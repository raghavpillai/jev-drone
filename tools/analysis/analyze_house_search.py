"""Inspect recorded flight phases and camera headings; evaluator output only."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def analyze(path, out):
    result = json.loads(path.read_text())
    calls = result["calls"]
    # The append-only journal is the canonical request state.
    journal = path.with_name("calls.jsonl")
    for call, raw in zip(calls, map(json.loads, journal.read_text().splitlines())):
        call["state"] = raw["state"]
    objective = result["world"]["mission"][0]
    entered = next(
        (c["time"] for c in calls if c["state"]["sensors"]["room"] == objective["room"]), None
    )
    seen = next(
        (
            c["time"]
            for c in calls
            if c["state"]["sensors"]["room"] == objective["room"]
            and any(d["marker"] == objective["marker"] for d in c["state"]["sensors"]["detections"])
        ),
        None,
    )
    inspected = next(
        (r["time"] for r in result["reports"] if r.get("stage") == 0 and r["valid"]), None
    )
    trace = result["trace"]
    times = np.array([t["time"] for t in trace])
    positions = np.array([t["position"] for t in trace])
    speeds = np.array([np.linalg.norm(t["velocity"]) for t in trace])
    phases = []
    milestones = [
        (0, "Navigate to room"),
        (entered, "Search"),
        (seen, "Approach"),
        (inspected, "Return"),
    ]
    milestones = [(t, label) for t, label in milestones if t is not None]
    colors = ["#40799e", "#c88931", "#32916e", "#94629c"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for box in result["world"]["geometry"]:
        if box["name"] in ("floor", "ceiling", "lintel"):
            continue
        center, half = np.array(box["center"]), np.array(box["half"])
        ax.add_patch(
            Rectangle(
                center[:2] - half[:2],
                2 * half[0],
                2 * half[1],
                facecolor=box["color"][:3],
                edgecolor="#777777",
                alpha=0.3,
                linewidth=0.3,
            )
        )
    for room, (lo, hi) in result["world"]["rooms"].items():
        ax.text((lo[0] + hi[0]) / 2, hi[1] - 0.3, room, ha="center", va="top", fontsize=9)
    for index, (start, label) in enumerate(milestones):
        end = milestones[index + 1][0] if index + 1 < len(milestones) else result["sim_seconds"]
        mask = (times >= start) & (times <= end)
        phase_positions = positions[mask]
        color = colors[index]
        ax.plot(*phase_positions[:, :2].T, color=color, label=label, linewidth=1.5)
        dt = np.diff(times, prepend=times[0])
        phases.append(
            {
                "phase": label,
                "start": start,
                "end": end,
                "duration": end - start,
                "path_metres": float(
                    np.linalg.norm(np.diff(phase_positions, axis=0), axis=1).sum()
                ),
                "stationary_seconds": float(dt[mask & (speeds < 0.05)].sum()),
            }
        )
        axes[1].barh(label, end - start, color=color)
    lo = np.min([r[0] for r in result["world"]["rooms"].values()], axis=0)
    hi = np.max([r[1] for r in result["world"]["rooms"].values()], axis=0)
    ax.set(
        xlim=(lo[0] - 0.3, hi[0] + 0.3),
        ylim=(lo[1] - 0.3, hi[1] + 0.3),
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.34, 0.06),
        ncol=2,
        fontsize=9,
    )
    axes[1].set(xlabel="Simulated seconds")
    axes[1].invert_yaxis()
    fig.suptitle(
        f"{path.parent.parent.name} / {result['id']} — {result['status']} ({result['stage']}/{len(result['world']['mission'])})"
    )
    fig.text(
        0.03,
        0.02,
        "Native trajectory over evaluator furniture geometry. Jev receives known room architecture and sensed obstacles, not this furniture map.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "phases.png", dpi=180)
    plt.close(fig)
    searches = []
    for task in result["tasks"]:
        if not task["name"].startswith("view_") or task.get("mission_stage", 0) != 0:
            continue
        end = task.get("ended", result["sim_seconds"])
        observations = [c["state"]["sensors"] for c in calls if task["time"] <= c["time"] <= end]
        searches.append(
            {
                "name": task["name"],
                "time": task["time"],
                "ended": end,
                "outcome": task["outcome"],
                "goal": task["position"],
                "look_position": task.get("look_position"),
                "final_position": task.get("end_position"),
                "final_heading": observations[-1]["yaw_degrees"] if observations else None,
                "final_altitude_error_m": abs(task["end_position"][2] - task["position"][2])
                if task.get("end_position")
                else None,
                "altitude_fraction_scope": "Request snapshots only; excludes a task-ending observation that issued no model request.",
                "requested_altitude_unreached_fraction": float(
                    np.mean(
                        [
                            abs(o["position_estimate"][2] - task["position"][2]) > 0.15
                            for o in observations
                        ]
                    )
                )
                if observations
                else None,
            }
        )
    immediate = []
    longest = []
    current = []
    for task in result["tasks"]:
        short_look = (
            task["name"].startswith("look_")
            and "position" not in task
            and task["outcome"] == "arrived"
            and task.get("ended", result["sim_seconds"]) - task["time"] <= 0.5
            and np.linalg.norm(np.array(task.get("end_position", task["start"])) - task["start"])
            <= 0.05
        )
        if short_look:
            immediate.append(task)
            current = (
                current + [task] if current and current[-1]["name"] == task["name"] else [task]
            )
            if len(current) > len(longest):
                longest = current
        else:
            current = []
    summary = {
        "source": str(path),
        "entered_target_room": entered,
        "first_target_seen": seen,
        "inspected": inspected,
        "phases": phases,
        "search_tasks": searches,
        "immediate_completed_looks": len(immediate),
        "longest_immediate_look_streak": {
            "count": len(longest),
            "name": longest[0]["name"] if longest else None,
            "start": longest[0]["time"] if longest else None,
            "end": longest[-1]["ended"] if longest else None,
        },
    }
    (out / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    analyze(a.episode, a.out)
