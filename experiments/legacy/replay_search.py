"""Animate a recorded room search, including camera aim and observed-object memory."""

import argparse
import bisect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=6.0)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    data = json.loads(args.episode.read_text())
    trace = data["trace"]
    times = [p[0] for p in trace]
    views = data["views"]
    view_times = [v["time"] for v in views]
    commands = []
    for call in data["calls"]:
        if call["role"] == "control":
            commands.append(
                (
                    call["sent_at"] + call["latency_seconds"],
                    call.get("selected_action", call["action"])
                    if call.get("accepted")
                    else "brake / stale response",
                )
            )
        else:
            commands.append((call["sent_at"], "brake / deliberation"))
    commands.sort()
    received = [t for t, _ in commands]
    fig, (plan, alt) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [4, 1]})
    fig.subplots_adjust(top=0.81, bottom=0.16, hspace=0.45)
    fig.patch.set_facecolor("#f5f2ea")
    furniture = []
    for obj in data["ground_truth"]["objects"]:
        lo, hi = obj["box"]["lo"], obj["box"]["hi"]
        patch = Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], facecolor="#adb4ab", alpha=0.12)
        plan.add_patch(patch)
        furniture.append((obj, patch))
    for wall in data["world"]["obstacles"][:4]:
        lo, hi = wall["lo"], wall["hi"]
        plan.add_patch(
            Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], facecolor="#737d70", alpha=0.5)
        )
    for x, name in ((3, "living room"), (9, "bedroom"), (15, "study")):
        plan.text(x, 9.7, name, ha="center", fontsize=9)
    plan.set(xlim=(0, 18), ylim=(0, 10), aspect="equal", xlabel="x / m", ylabel="y / m")
    (path,) = plan.plot([], [], color="#247b6e", linewidth=1.5)
    drone = Rectangle((0, 0), 0.4, 0.4, facecolor="#164e47", edgecolor="white", zorder=5)
    plan.add_patch(drone)
    (aim,) = plan.plot([], [], color="#825f9c", linewidth=2)
    (height,) = alt.plot([], [], color="#247b6e", linewidth=1.5)
    alt.set(xlim=(0, times[-1]), ylim=(0, 4), xlabel="Simulation seconds", ylabel="Height / m")
    title = fig.suptitle("", fontsize=12, x=0.1, ha="left")
    caption = fig.text(0.1, 0.035, "", fontsize=9)
    directions = {
        "forward": (1, 0),
        "back": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
        "up": (0, 0),
        "down": (0, 0),
    }
    # Derive persistent memory from actual logged sightings, not target labels.
    seen = set()
    memory = []
    for view in views:
        seen.update(view["objects"])
        memory.append(seen.copy())

    def update(t):
        index = max(0, bisect.bisect_right(times, t) - 1)
        p = trace[index]
        prefix = trace[: index + 1]
        view_index = max(0, bisect.bisect_right(view_times, t) - 1)
        view = views[view_index]
        seen = memory[view_index]
        path.set_data([r[1] for r in prefix], [r[2] for r in prefix])
        height.set_data([r[0] for r in prefix], [r[3] for r in prefix])
        drone.set_xy((p[1] - 0.2, p[2] - 0.2))
        dx, dy = directions[view["look"]]
        aim.set_data([p[1], p[1] + dx], [p[2], p[2] + dy])
        for obj, patch in furniture:
            patch.set_alpha(0.55 if obj["id"] in seen else 0.12)
            patch.set_facecolor(
                "#d9843b"
                if obj["id"] in seen and obj["id"] in data["eligible_target_ids"]
                else "#adb4ab"
            )
        ci = bisect.bisect_right(received, t) - 1
        action = "brake / deliberation"
        if ci >= 0 and t - received[ci] <= data["physics"]["command_lease_s"]:
            action = commands[ci][1]
        status = data["status"].replace("_", " ") if t >= times[-1] else "searching"
        title.set_text(
            f"Jev room search · {data['item']} in {data['room'].replace('_', ' ')}\n"
            f"{status} | {t:.1f}s | {args.speed:g}× playback | height {p[3]:.2f}m\n"
            f"Action: {action} | Camera: {view['look']} | Objects remembered: {len(seen)}"
        )
        caption.set_text(
            "Recorded live OpenRouter decisions. Purple line: camera direction in the floor plane.\n"
            "Faint boxes: hidden evaluator geometry; solid boxes: previously observed; orange: observed target.\n"
            "Synthetic sensing and stabilized translation; this is not a camera image or rotor simulation."
        )
        return []

    frames = [i * args.speed / 10 for i in range(int(times[-1] * 10 / args.speed) + 1)] + [
        times[-1]
    ] * 12
    movie = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    movie.save(args.out, writer=animation.PillowWriter(fps=10), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
