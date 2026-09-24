"""Replay measured camera frames beside an explicitly evaluator-only map."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def replay(path, out):
    result = json.loads(path.read_text())
    trace = result["trace"]
    frame_dir = path.with_name(path.stem + "-frames")
    snapshots = sorted(frame_dir.glob("*.npz"))
    if not snapshots:
        raise ValueError("Episode has no saved camera frames")
    snapshot_times = [float(p.stem) for p in snapshots]
    images = []
    rooms = result["world"]["rooms"]
    lower = np.min([bounds[0] for bounds in rooms.values()], axis=0)
    upper = np.max([bounds[1] for bounds in rooms.values()], axis=0)
    scale = min(588 / (upper[0] - lower[0]), 392 / (upper[1] - lower[1]))

    def xy(point):
        return (40 + (point[0] - lower[0]) * scale, 495 - (point[1] - lower[1]) * scale)

    for time in np.linspace(
        0, result["sim_seconds"], min(100, max(2, int(result["sim_seconds"] * 1.4)))
    ):
        index = max(0, int(np.searchsorted([t["time"] for t in trace], time, side="right")) - 1)
        point = trace[index]
        canvas = Image.new("RGB", (1100, 640), "#10171f")
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 16), result["id"], font=font(24), fill="white")
        draw.text(
            (28, 52),
            "Evaluator map · hidden geometry is NOT sent to Jev",
            font=font(14),
            fill="#a7b8c9",
        )
        for box in result["world"]["geometry"]:
            if box["name"] in ("floor", "ceiling", "lintel") or box["dynamic"]:
                continue
            center, half = np.array(box["center"]), np.array(box["half"])
            a, b = xy(center - half), xy(center + half)
            color = tuple(int(c * 165) for c in box["color"][:3])
            draw.rectangle((a[0], b[1], b[0], a[1]), fill=color, outline="#748190")
        for name, (lo, hi) in rooms.items():
            x, y = xy(((lo[0] + hi[0]) / 2, hi[1] - 0.3))
            draw.text((x, y), name, anchor="mt", font=font(12), fill="#d7e1eb")
        for actor in point["actors"]:
            x, y = xy(actor)
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="#51ce84")
        route = [xy(t["position"]) for t in trace[: index + 1]]
        if len(route) > 1:
            draw.line(route, fill="#55d9e8", width=3)
        x, y = xy(point["position"])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="white")
        look = point["camera"]
        draw.line((x, y, x + 25 * look[0], y - 25 * look[1]), fill="#f4da80", width=3)
        snapshot_index = max(0, int(np.searchsorted(snapshot_times, time, side="right")) - 1)
        with np.load(snapshots[snapshot_index]) as data:
            camera_index = 0
            if result["world"]["config"].get("front_controls") and "camera_names" in data:
                indices = np.flatnonzero(data["camera_names"] == "east")
                camera_index = int(indices[0]) if len(indices) else None
            rgb = (
                data["rgb"][camera_index]
                if camera_index is not None
                else np.zeros_like(data["rgb"][0])
            )
            depth = (
                data["depth"][camera_index]
                if camera_index is not None
                else np.full_like(data["depth"][0], np.nan)
            )
        canvas.paste(Image.fromarray(rgb).resize((384, 288), Image.Resampling.NEAREST), (686, 102))
        caption = (
            "Front camera only · sampled every 2s"
            if result["world"]["config"].get("front_controls")
            else "Rendered RGB · sampled every 2s"
        )
        draw.text((686, 76), caption, font=font(15), fill="#a7b8c9")
        finite = np.isfinite(depth)
        shade = np.nan_to_num(np.clip(depth / 8.0, 0, 1))
        depth_rgb = np.stack([shade * 120, (1 - shade) * 180, (1 - shade) * 230], axis=-1).astype(
            np.uint8
        )
        depth_rgb[~finite] = (225, 60, 120)
        canvas.paste(
            Image.fromarray(depth_rgb).resize((192, 144), Image.Resampling.NEAREST), (686, 420)
        )
        draw.text((888, 420), "Depth", font=font(17), fill="white")
        draw.text((888, 449), "Pink = missing", font=font(13), fill="#a7b8c9")
        if result["world"]["config"].get("control_schema") == "joystick-v1":
            epoch = result["world"]["mission_start_sim_time"]
            published = [
                s
                for s in result.get("setpoints", [])
                if s["sim"] - epoch <= time and s.get("body_controls")
            ]
            values = published[-1]["body_controls"] if published else {}
            draw.text((888, 483), "Held controls", font=font(15), fill="white")
            for row, (key, label, unit) in enumerate(
                (
                    ("forward_mps", "Forward", "m/s"),
                    ("right_mps", "Right", "m/s"),
                    ("up_mps", "Up", "m/s"),
                    ("yaw_rate_rps", "Pan left", "rad/s"),
                )
            ):
                draw.text(
                    (888, 508 + row * 20),
                    f"{label}: {values.get(key, 0.0):+.2f} {unit}",
                    font=font(13),
                    fill="#55d9e8",
                )
        calls = [c for c in result["calls"] if c.get("response_time", c["time"]) <= time]
        latest = calls[-1] if calls else {}
        draw.text(
            (30, 535),
            f"t={time:.1f}s   altitude={point['position'][2]:.2f}m   speed={np.linalg.norm(point['velocity']):.2f}m/s",
            font=font(18),
            fill="white",
        )
        draw.text(
            (30, 571),
            f"Jev: {latest.get('choice', 'waiting')}  ({latest.get('role', 'sensors')})",
            font=font(18),
            fill="#55d9e8",
        )
        draw.text(
            (30, 606),
            f"Final: {result['status']} · contacts: {len(result['contacts'])} · rule violations: {len(result.get('violations', []))} · Jev / OpenRouter navigation",
            font=font(14),
            fill="#a7b8c9",
        )
        images.append(canvas)
    images[-1].save(out.with_suffix(".png"))
    images[0].save(out, save_all=True, append_images=images[1:], duration=100, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay(args.episode, args.out)


if __name__ == "__main__":
    main()
