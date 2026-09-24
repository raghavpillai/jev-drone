"""Summarize all architecture trials, including failures, without model calls."""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def metrics(path, e):
    controls = [c for c in e["calls"] if c["role"] == "control"]
    accepted = sum(c["accepted"] for c in controls)
    roles = {
        role: [c["latency_seconds"] for c in e["calls"] if c["role"] == role]
        for role in ["planner", "navigator", "recognition", "control"]
    }
    return {
        "run": path.parent.name,
        "stage": "confirm" if "confirm" in path.parent.name else "screen",
        "architecture": e["architecture"],
        "planner": e["planner"],
        "case": e["case"],
        "seed": e["seed"],
        "status": e["status"],
        "seconds": e["simulation_seconds"],
        "path_m": e["path_metres"],
        "first_sighting_seconds": e["target_first_seen_seconds"],
        "coverage": e["coverage"],
        "calls": len(e["calls"]),
        "role_counts": e["role_counts"],
        "cost_usd": e["cost_usd"],
        "accepted_movement_hz": accepted / max(e["simulation_seconds"], 0.001),
        "movement_active_hz": accepted
        / max(sum(max(0.2, c["latency_seconds"]) for c in controls), 0.001),
        "latency_median_ms": {
            r: 1000 * statistics.median(xs) if xs else None for r, xs in roles.items()
        },
        "latency_p95_ms": {
            r: 1000 * sorted(xs)[min(len(xs) - 1, int(len(xs) * 0.95))] if xs else None
            for r, xs in roles.items()
        },
        "request_seconds_by_role": {r: sum(xs) for r, xs in roles.items()},
        "errors": sum(bool(c.get("error")) for c in e["calls"]),
        "stale_control_calls": sum(c["latency_seconds"] > 0.8 for c in controls),
        "movement_models": e["movement_models_returned"],
    }


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["architecture"], row["planner"])].append(row)
    result = []
    for (stage, architecture, planner), group in sorted(grouped.items()):
        present = [r for r in group if r["case"] != "absent"]
        successes = [r for r in present if r["status"] == "success"]
        result.append(
            {
                "stage": stage,
                "architecture": architecture,
                "planner": planner,
                "present_successes": len(successes),
                "present_trials": len(present),
                "absence_outcomes": [r["status"] for r in group if r["case"] == "absent"],
                "collisions": sum(r["status"] == "collision" for r in group),
                "statuses": dict(Counter(r["status"] for r in group)),
                "success_median_seconds": statistics.median(r["seconds"] for r in successes)
                if successes
                else None,
                "calls": sum(r["calls"] for r in group),
                "cost_usd": sum(r["cost_usd"] for r in group),
                "planner_calls": sum(r["role_counts"].get("planner", 0) for r in group),
            }
        )
    return result


def plot(rows, stage):
    rows = [r for r in rows if r["stage"] == stage]
    conditions = sorted({(r["architecture"], r["planner"]) for r in rows})
    if not conditions:
        return
    fig, ax = plt.subplots(figsize=(11, 1.6 + len(conditions) * 1.15), layout="constrained")
    cases = ["near", "far", "occluded", "absent"]
    colors = {
        "success": "#c9e9dc",
        "correct_absence": "#c9e9dc",
        "timeout": "#f4e4b7",
        "collision": "#f2c9c9",
    }
    lookup = {(r["architecture"], r["planner"], r["case"]): r for r in rows}
    for y, (architecture, planner) in enumerate(conditions):
        for x, case in enumerate(cases):
            row = lookup.get((architecture, planner, case))
            if row:
                label = f"{row['status'].replace('_', ' ')}\n{row['seconds']:.1f} s · {row['role_counts'].get('planner', 0)} planner calls"
                color = colors.get(row["status"], "#e6d4ef")
            else:
                label, color = "pending", "#eeeeee"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=3))
            ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=10)
    ax.set(
        xlim=(0, 4),
        ylim=(len(conditions), 0),
        xticks=[i + 0.5 for i in range(4)],
        xticklabels=["Nearby target", "Target in third room", "Occluded target", "Target absent"],
        yticks=[i + 0.5 for i in range(len(conditions))],
        yticklabels=[f"{a.title()} handoff\n{p.title()} planner" for a, p in conditions],
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"{stage.title()} trials: every movement decision uses Jev\nMatched scenes · actual API delay included · 210 s deadline",
        pad=34,
        fontsize=14,
    )
    fig.savefig(f"report/architecture-{stage}.png", dpi=150)
    plt.close(fig)


def main():
    paths = sorted(Path("results").glob("architecture-*/episode.json"))
    rows = [metrics(p, json.loads(p.read_text())) for p in paths]
    summary = aggregate(rows)
    Path("report/architecture-results.json").write_text(
        json.dumps({"episodes": rows, "summary": summary}, indent=2)
    )
    for stage in ["screen", "confirm"]:
        plot(rows, stage)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
