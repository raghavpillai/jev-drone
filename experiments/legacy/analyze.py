"""Produce a compact measured-results table and trajectory figure from saved runs."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def percentile(values, fraction):
    if not values:
        return 0.
    values = sorted(values)
    index = (len(values)-1)*fraction
    lo = int(index)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(index-lo)


def read_runs(root):
    runs = defaultdict(list)
    for path in sorted(root.glob("*/*.json")):
        if path.name == "metadata.json":
            continue
        data = json.loads(path.read_text())
        if "world" in data:
            runs[path.parent.name].append(data)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("report"))
    parser.add_argument("--plot-runs", nargs="+", help="Optional run directories to include in the figure; summaries always include all runs")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    runs = read_runs(args.results)
    rows = []
    for name, episodes in runs.items():
        times = [d["latency_seconds"]*1000 for e in episodes for d in e["decisions"]] if episodes[0]["policy"] == "jev" else []
        rows.append({"run": name, "episodes": len(episodes),
                     "success": sum(e["status"] == "success" for e in episodes),
                     "collision": sum(e["status"] == "collision" for e in episodes),
                     "timeout": sum(e["status"] == "timeout" for e in episodes),
                     "other": sum(e["status"] not in ["success", "collision", "timeout"] for e in episodes),
                     "calls": sum(e["calls"] for e in episodes),
                     "median_ms": round(statistics.median(times), 1) if times else "",
                     "p95_ms": round(percentile(times, .95), 1) if times else "",
                     "cost_usd": round(sum(e["cost_usd"] for e in episodes), 6),
                     "errors": sum(e["errors"] for e in episodes),
                     "stale": sum(e["stale_decisions"] for e in episodes)})
    if not rows:
        raise SystemExit("No completed episodes found")
    with (args.out/"summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = "| Run | Success | Collision | Timeout | Other | Calls | Median / p95 latency | Cost |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    table = header + "\n".join(
        f"| {r['run']} | {r['success']}/{r['episodes']} | {r['collision']} | {r['timeout']} | {r['other']} | {r['calls']} | {r['median_ms']} / {r['p95_ms']} ms | ${r['cost_usd']:.6f} |" for r in rows)
    (args.out/"results.md").write_text("# Measured results\n\n"+table+"\n\nSynthetic navigation with measured API delays, except explicitly labeled lockstep runs. Calibration layouts are not held-out tests. Baselines use fixed delays, not identical live latency traces. No camera perception, wind, attitude, or rotor dynamics.\n")
    print(table)
    plot_rows = []
    for name, episodes in runs.items():
        if episodes[0]["policy"] != "jev":
            continue
        if args.plot_runs and name not in args.plot_runs:
            continue
        for seed in sorted({e["world"]["seed"] for e in episodes}):
            plot_rows.append((f"{name} / seed {seed}", {e["world"]["name"]: e for e in episodes if e["world"]["seed"] == seed}))
    if not plot_rows:
        return
    names = ["open", "pillar", "wall", "overpass", "u_trap"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(len(plot_rows), 5, figsize=(16, 3.2*len(plot_rows)), squeeze=False, constrained_layout=True)
    colors = {"success": "#14845d", "collision": "#c54145", "timeout": "#b67918"}
    for row, (label, episodes) in enumerate(plot_rows):
        for column, name in enumerate(names):
            ax = axes[row, column]
            if name not in episodes:
                ax.axis("off")
                continue
            e = episodes[name]
            world = e["world"]
            vertical = name == "overpass"
            a, b = (0, 2) if vertical else (0, 1)
            for box in world["obstacles"]:
                lo, hi = box["lo"], box["hi"]
                ax.add_patch(Rectangle((lo[a], lo[b]), hi[a]-lo[a], hi[b]-lo[b], color="#d8dce0"))
            trace = e["trace"]
            color = colors.get(e["status"], "#666666")
            ax.plot([p[a+1] for p in trace], [p[b+1] for p in trace], color=color, linewidth=2)
            ax.scatter(world["start"][a], world["start"][b], s=32, color="#24384d", zorder=5)
            ax.scatter(world["goal"][a], world["goal"][b], marker="*", s=140, color="#1970b5", zorder=5)
            ax.scatter(trace[-1][a+1], trace[-1][b+1], marker="x", s=38, color=color, zorder=6)
            ax.set_xlim(0, world["size"][a])
            ax.set_ylim(0, world["size"][b])
            ax.set_aspect("equal")
            ax.set_title(f"{name.replace('_', ' ')} · {e['status']}\n{e['simulation_seconds']:.1f}s / {e['final_distance_metres']:.2f}m remaining", loc="left", fontsize=10)
            ax.set_xlabel("x / metres")
            ax.set_ylabel("altitude / metres" if vertical else "y / metres")
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.text(0, 1.3, label, transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.suptitle("Jev navigation prototype — actual recorded trajectories\nStart ●   Goal ★   End ×   Gray: obstacles   Overpass shown from the side", fontsize=16)
    fig.savefig(args.out/"trajectories.png", dpi=140, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
