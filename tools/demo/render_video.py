"""Render every search-replay frame independently, then verify the encoded FPS."""

import argparse
import base64
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright


def render(output, url, speed):
    root = Path(__file__).resolve().parents[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    errors, snapshots, frame_hashes = [], [], []
    started = time.monotonic()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        page.wait_for_function("window.stateReplay?.ready")
        status = page.evaluate("stateReplay.status")
        end = status["duration"]
        assert status["wallHeight"] == 2.25 and status["targetHighlighted"]
        assert not page.locator("button,input,select").count()
        assert page.locator("#mission-objective").inner_text() == "Find the red item"
        footer = page.locator(".state-rail footer").bounding_box()
        assert footer["y"] + footer["height"] <= 1080
        mission = page.locator(".mission-brief").bounding_box()
        status_card = page.locator(".mission").bounding_box()
        assert mission["x"] >= 1488 and mission["y"] < status_card["y"]
        assert (
            page.locator(".mission").evaluate("(element) => element.parentElement.className")
            == "state-rail"
        )
        source = json.loads((root / "demo/data/state-replay.json").read_text())
        episode = json.loads((root / "demo" / source["episode"]).read_text())
        objective_stage = next(
            i for i, objective in enumerate(episode["world"]["mission"]) if objective.get("marker")
        )
        found = any(
            report["valid"] and report["stage"] == objective_stage for report in episode["reports"]
        )
        # A dropped render cannot shorten this loop: output timestamps come from
        # the explicit frame number, never browser wall time or screen capture.
        flight_frames = math.ceil(end / speed * fps) + 1
        frames = flight_frames + 3 * fps
        cdp = page.context.new_cdp_session(page)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            str(fps),
            "-vcodec",
            "mjpeg",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-movflags",
            "+faststart",
            str(output),
        ]
        with output.with_suffix(".encode.log").open("w") as log:
            encoder = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log
            )
            try:
                for index in range(frames):
                    rendered = page.evaluate(
                        "([i,fps,speed])=>stateReplay.renderFrame(i,fps,speed)", [index, fps, speed]
                    )
                    assert math.isclose(
                        rendered["time"], min(end, index * speed / fps), abs_tol=1e-9
                    )
                    screenshot = cdp.send(
                        "Page.captureScreenshot",
                        {
                            "format": "jpeg",
                            "quality": 97,
                            "fromSurface": True,
                            "captureBeyondViewport": False,
                            "optimizeForSpeed": True,
                        },
                    )
                    frame = base64.b64decode(screenshot["data"])
                    encoder.stdin.write(frame)
                    frame_hashes.append(hashlib.sha256(frame).hexdigest())
                    if index % 150 == 0 or index == frames - 1:
                        print(
                            json.dumps(
                                {
                                    "frame": index + 1,
                                    "frames": frames,
                                    "sim_seconds": round(rendered["time"], 2),
                                    "render_wall_seconds": round(time.monotonic() - started),
                                }
                            ),
                            flush=True,
                        )
                        snapshots.append(rendered)
                    if index in (0, flight_frames // 2, flight_frames - 1):
                        label = (
                            "start"
                            if index == 0
                            else ("found" if found else "unfinished")
                            if index == flight_frames - 1
                            else "search"
                        )
                        output.with_name(output.stem + "-" + label + ".jpg").write_bytes(frame)
                    if errors:
                        raise RuntimeError(errors)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError(output.with_suffix(".encode.log").read_text())
            except BaseException:
                encoder.kill()
                encoder.wait()
                raise
        assert page.locator("#mission-phase").inner_text() == (
            "Item found" if found else "Search unfinished"
        )
        browser.close()

    # Decode the actual MP4, not just its nominal container rate. Each source
    # frame must survive, with one 1/30-second timestamp and no repeated frames.
    checksum_file = output.with_suffix(".framemd5")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(output),
            "-an",
            "-f",
            "framemd5",
            str(checksum_file),
        ],
        check=True,
    )
    checksum_text = checksum_file.read_text()
    assert "#tb 0: 1/30" in checksum_text
    decoded = [
        line.split(",")
        for line in checksum_text.splitlines()
        if not line.startswith("#") and line.strip()
    ]
    assert len(decoded) == frames
    assert all(int(row[2]) == index and int(row[3]) == 1 for index, row in enumerate(decoded))
    repeated = sum(a[-1].strip() == b[-1].strip() for a, b in zip(decoded, decoded[1:]))
    assert repeated == 0, f"{repeated} repeated decoded frames"
    reader = imageio_ffmpeg.read_frames(str(output))
    metadata = next(reader)
    reader.close()
    assert metadata["size"] == (1920, 1080) and metadata["fps"] == fps
    manifest = {
        "file": str(output),
        "source_episode": source["episode"],
        "request_provenance": source["provenance"],
        "scope": "Recorded search through verified inspection or the run deadline. Source flight and request journal are identified in provenance; any later objectives are excluded.",
        "item_found": found,
        "viewer_effects": "Oak/fabric furniture finishes, 2.25 m cutaway walls, target emission and sparkles. Original sensor frames and control decisions unchanged.",
        "render_method": "Independent fixed-time browser renders piped directly to H.264, not real-time screen recording.",
        "resolution": metadata["size"],
        "fps": fps,
        "frames": frames,
        "flight_frames": flight_frames,
        "duration_seconds": frames / fps,
        "playback_rate": speed,
        "mission_end_sim_seconds": end,
        "repeated_decoded_frames": repeated,
        "unique_capture_frames": len(set(frame_hashes)),
        "full_decode_verified": True,
        "browser_errors": errors,
        "size_bytes": output.stat().st_size,
        "render_wall_seconds": time.monotonic() - started,
        "snapshots": snapshots,
        "renderer_sha256": {
            name: hashlib.sha256((root / "demo" / name).read_bytes()).hexdigest()
            for name in (
                "src/state.js",
                "state.html",
                "src/state.css",
                "src/scene.js",
                "src/furnishings.js",
                "src/target-effects.js",
            )
        },
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in manifest.items()
                if k not in ("snapshots", "renderer_sha256", "request_provenance")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("report/px4-find-video/jev-best-mission-12x.mp4")
    )
    parser.add_argument("--url", default="http://127.0.0.1:8765/state.html?render")
    parser.add_argument("--speed", type=float, default=12)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    render(args.out.resolve(), args.url, args.speed)
