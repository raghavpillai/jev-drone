"""Summarize the recorded structured-state experiments, including failures."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    rows, groups = [], defaultdict(list)
    for folder in sorted(Path("results").glob("representation-*")):
        for path in sorted(folder.glob("*.json")):
            d = json.loads(path.read_text())
            if "status" not in d:
                continue
            calls = d["decisions"]
            row = {
                k: d[k]
                for k in (
                    "variant",
                    "case",
                    "seed",
                    "degraded",
                    "status",
                    "calls",
                    "cost_usd",
                    "simulation_seconds",
                )
            }
            row.update(
                source=str(path),
                accepted_hz=sum(c["accepted"] for c in calls) / d["simulation_seconds"],
                latency_median_ms=1000 * statistics.median(c["latency_seconds"] for c in calls),
                errors=sum(bool(c.get("error")) for c in calls),
                stale=sum(c["latency_seconds"] > 0.8 for c in calls),
                prompt_tokens=sum(c.get("usage", {}).get("prompt_tokens", 0) or 0 for c in calls),
                models=sorted({c["model"] for c in calls if c.get("model")}),
            )
            rows.append(row)
            groups[d["variant"]].append(row)
    summary = {}
    for variant, items in groups.items():
        summary[variant] = {
            "episodes": len(items),
            "success": sum(r["status"] == "success" for r in items),
            "collision": sum(r["status"] == "collision" for r in items),
            "timeout": sum(r["status"] == "timeout" for r in items),
            "calls": sum(r["calls"] for r in items),
            "cost_usd": sum(r["cost_usd"] for r in items),
            "median_episode_hz": statistics.median(r["accepted_hz"] for r in items),
            "median_episode_latency_ms": statistics.median(r["latency_median_ms"] for r in items),
        }
    output = Path("report")
    output.mkdir(exist_ok=True)
    (output / "representation-results.json").write_text(
        json.dumps({"summary": summary, "episodes": rows}, indent=2)
    )
    variants = [v for v in ("numeric", "facts", "grid", "brief") if v in groups]
    conditions = [(seed, degraded) for seed in (0, 1) for degraded in (False, True)]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    colors = {"success": "#c1e6d0", "collision": "#f4bcbc", "timeout": "#f7e3b0"}
    cells, shades = [], []
    for variant in variants:
        labels, backgrounds = [], []
        for seed, degraded in conditions:
            items = [r for r in groups[variant] if r["seed"] == seed and r["degraded"] == degraded]
            wins = sum(r["status"] == "success" for r in items)
            crashes = sum(r["status"] == "collision" for r in items)
            labels.append(
                f"{wins}/{len(items)} completed\n{crashes} collisions" if items else "Not tested"
            )
            backgrounds.append(
                "#eeeeee"
                if not items
                else colors["success"]
                if wins == len(items)
                else colors["timeout"]
                if wins
                else colors["collision"]
            )
        cells.append(labels)
        shades.append(backgrounds)
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        cellColours=shades,
        rowLabels=variants,
        colLabels=[
            "Original · clean",
            "Original · degraded",
            "Changed · clean",
            "Changed · degraded",
        ],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.9)
    ax.set_title(
        "Jev local control: same observations, different representations", fontsize=15, pad=24
    )
    fig.text(
        0.5,
        0.025,
        "Four courses per cell · 40 s limit · actual API delay · degraded = missing + noisy + delayed ranges",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.07, 0.07, 1, 1))
    fig.savefig(output / "representation-results.png", dpi=170)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
