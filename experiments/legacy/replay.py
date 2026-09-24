"""Render a saved episode as a GIF. No network calls or model inference."""

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
    parser.add_argument("--speed", type=float, default=3.0)
    args = parser.parse_args()
    episode = json.loads(args.episode.read_text())
    world = episode["world"]
    trace = episode["trace"]
    times = [p[0] for p in trace]
    decisions = [d for d in episode["decisions"] if d["accepted"]]
    received = [d["received_at"] for d in decisions]
    color = "#14845d" if episode["status"] == "success" else "#c54145"
    fig, (top, side) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(top=0.79, bottom=0.16, wspace=0.3)
    artists = []
    for ax, vertical in [(top, 1), (side, 2)]:
        for box in world["obstacles"]:
            lo, hi = box["lo"], box["hi"]
            ax.add_patch(
                Rectangle(
                    (lo[0], lo[vertical]),
                    hi[0] - lo[0],
                    hi[vertical] - lo[vertical],
                    color="#d8dce0",
                )
            )
        ax.scatter(
            world["start"][0], world["start"][vertical], s=35, color="#24384d", label="Start"
        )
        ax.scatter(
            world["goal"][0],
            world["goal"][vertical],
            marker="*",
            s=160,
            color="#1970b5",
            label="Goal",
        )
        ax.set(
            xlim=(0, 12),
            ylim=(0, world["size"][vertical]),
            xlabel="x / metres",
            ylabel="y / metres" if vertical == 1 else "altitude / metres",
            title="Top view" if vertical == 1 else "Side projection (obstacle depth omitted)",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)
        (path,) = ax.plot([], [], color=color, linewidth=1.8)
        drone = Rectangle((0, 0), 0.4, 0.4, facecolor=color, edgecolor="white", zorder=5)
        ax.add_patch(drone)
        artists.append((path, drone, vertical))
    title = fig.suptitle("", fontsize=14, x=0.08, ha="left")
    caption = fig.text(0.08, 0.055, "", fontsize=10)
    frame_times = [i * args.speed / 15 for i in range(int(times[-1] * 15 / args.speed) + 1)] + [
        times[-1]
    ] * 15

    def update(t):
        index = min(bisect.bisect_right(times, t), len(trace))
        point = trace[max(0, index - 1)]
        for path, drone, axis in artists:
            path.set_data([p[1] for p in trace[:index]], [p[axis + 1] for p in trace[:index]])
            drone.set_xy((point[1] - 0.2, point[axis + 1] - 0.2))
        decision_index = bisect.bisect_right(received, t) - 1
        decision = decisions[decision_index] if decision_index >= 0 else None
        action = "brake (waiting)"
        if decision:
            action = (
                decision["action"]
                if t - decision["received_at"] < episode["physics"]["command_lease_s"]
                else "brake (expired command)"
            )
        status = episode["status"] if t >= times[-1] else "running"
        title.set_text(
            f"Jev live-call replay / {world['name']} / seed {world['seed']}\n{status.upper()} · simulation {min(t, times[-1]):.1f}s · playback {args.speed:g}×"
        )
        detail = f"Model action: {action}"
        if decision:
            detail += f"    Stored maneuver: {decision.get('intent', 'none')}    Request: {decision['latency_seconds'] * 1000:.0f} ms"
        caption.set_text(
            detail
            + "\nRecorded trajectory with measured request delays. Simplified dynamics; no camera perception."
        )
        return []

    movie = animation.FuncAnimation(fig, update, frames=frame_times, interval=1000 / 15, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    movie.save(args.out, writer=animation.PillowWriter(fps=15), dpi=100)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
