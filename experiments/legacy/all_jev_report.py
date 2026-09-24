"""Summarize all-Jev experiments without pooling adaptive follow-ups into the screen."""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect():
    paths = list(Path("results/memory-screen").glob("*.json")) + list(
        Path("results/memory-limited").glob("*.json")
    )
    paths += list(Path("results").glob("all-jev-*/episode.json")) + list(
        Path("results").glob("inspection-*/episode.json")
    )
    paths += list(Path("results").glob("scanning-*/episode.json"))
    rows = []
    for path in sorted(paths):
        d = json.loads(path.read_text())
        if "status" not in d:
            continue
        calls = d["calls"]
        controls = [c for c in calls if c["role"] == "control"]
        if path.parent.name == "memory-screen":
            group = "Local memory / " + d["memory_mode"]
        elif path.parent.name == "memory-limited":
            group = "Limited depth / recovery"
        elif d.get("joint_scan_control"):
            group = "Joint scan follow-up / " + d["memory_mode"]
        elif d.get("inspection_context"):
            group = "Inspection follow-up / history"
        else:
            group = "Room search / " + d["memory_mode"]
        rows.append(
            {
                "source": str(path),
                "group": group,
                "case": d["case"],
                "seed": d["seed"],
                "status": d["status"],
                "seconds": d["simulation_seconds"],
                "calls": len(calls),
                "roles": dict(Counter(c["role"] for c in calls)),
                "cost_usd": d["cost_usd"],
                "accepted_movement_hz": sum(bool(c.get("accepted")) for c in controls)
                / d["simulation_seconds"],
                "control_latency_median_ms": 1000
                * statistics.median(c["latency_seconds"] for c in controls)
                if controls
                else None,
                "errors": sum(bool(c.get("error")) for c in calls),
                "stale_controls": sum(c["latency_seconds"] > 0.8 for c in controls),
                "returned_models": sorted({c["model"] for c in calls if c.get("model")}),
                "recovery_decisions": len(d.get("recovery_events", [])),
                "observed_map_cells": d.get("observed_map_cells", 0),
            }
        )
    return rows


def main():
    rows = collect()
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    summary = {}
    for group, items in groups.items():
        summary[group] = {
            "episodes": len(items),
            "outcomes": dict(Counter(r["status"] for r in items)),
            "calls": sum(r["calls"] for r in items),
            "cost_usd": sum(r["cost_usd"] for r in items),
            "median_episode_movement_hz": statistics.median(
                r["accepted_movement_hz"] for r in items
            ),
        }
    result = {
        "summary": summary,
        "episodes": rows,
        "total_calls": sum(r["calls"] for r in rows),
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
    }
    Path("report/all-jev-results.json").write_text(json.dumps(result, indent=2))
    names = list(summary)
    fig, ax = plt.subplots(figsize=(11, 5))
    left = [0] * len(names)
    for status, color in (
        ("success", "#45a77f"),
        ("correct_absence", "#73cba6"),
        ("timeout", "#e4b45f"),
        ("collision", "#d87b79"),
    ):
        counts = [summary[name]["outcomes"].get(status, 0) for name in names]
        ax.barh(names, counts, left=left, label=status, color=color)
        left = [a + b for a, b in zip(left, counts)]
    for i, name in enumerate(names):
        ax.text(
            summary[name]["episodes"] + 0.1,
            i,
            str(summary[name]["outcomes"]),
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(left, default=1) + 4)
    ax.set_xlabel("Episodes (different suites; compare within each suite)")
    ax.set_title("All decisions from Jev — memory, limited observation, and inspection")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig("report/all-jev-results.png", dpi=160)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
