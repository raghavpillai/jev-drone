"""Compare recorded trajectories; hidden furniture is evaluator display only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def main():
    report = Path("report/px4-full-missions")
    cohorts = [
        ("v36-paired-reference", "v36 reference", 5611),
        ("v39-validation", "v39 broad height facts", 5611),
        ("v40-scope", "v40 scoped height facts", 5711),
    ]
    colors = {
        "Navigate to room": "#376d93",
        "Search": "#ba812c",
        "Approach": "#258563",
        "Return": "#926391",
    }
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    for row, (cohort, label, hidden_seed) in enumerate(cohorts):
        for col, (case, seed) in enumerate(
            [("house_hidden", hidden_seed), ("house_reverse", 5612)]
        ):
            tag = f"{cohort}--{case}-{seed}"
            folder = report / tag
            data = json.loads(
                Path("results", f"px4-house-{cohort}", f"{case}-{seed}", "result.json").read_text()
            )
            analysis = json.loads((folder / "analysis.json").read_text())
            audit = json.loads((folder / "audit.json").read_text())
            ax = axes[row, col]
            for box in data["world"]["geometry"]:
                if box["name"] in ("floor", "ceiling", "lintel"):
                    continue
                center, half = np.array(box["center"]), np.array(box["half"])
                ax.add_patch(
                    Rectangle(
                        center[:2] - half[:2],
                        2 * half[0],
                        2 * half[1],
                        facecolor="#d9dcda",
                        edgecolor="#bcc2be",
                        alpha=0.55,
                        lw=0.35,
                    )
                )
            for name, (lo, hi) in data["world"]["rooms"].items():
                ax.text(
                    (lo[0] + hi[0]) / 2,
                    hi[1] - 0.3,
                    name,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#636d65",
                )
            times = np.array([s["time"] for s in data["trace"]])
            pos = np.array([s["position"] for s in data["trace"]])
            for phase in analysis["phases"]:
                mask = (times >= phase["start"]) & (times <= phase["end"])
                points = pos[mask]
                if len(points):
                    ax.plot(points[:, 0], points[:, 1], color=colors[phase["phase"]], lw=1.6)
            ax.scatter(*pos[0, :2], s=30, c="#22352a", zorder=8)
            ax.scatter(
                *pos[-1, :2],
                s=50,
                c="#267148" if audit["strict_pass"] else "#b84134",
                marker="s",
                zorder=8,
            )
            outcome = "strict pass" if audit["strict_pass"] else "strict failure"
            population = (
                " · fresh seed"
                if row == 2 and col == 0
                else " · regression repeat"
                if row == 2
                else ""
            )
            ax.set_title(
                f"{label} · {case.removeprefix('house_')} {seed}{population}\n{data['stage']}/{len(data['world']['mission'])} objectives · {outcome} · {data['sim_seconds']:.0f} s · {len(data['violations'])} clearance violations",
                fontsize=10,
                pad=10,
            )
            ax.set(
                xlim=(-0.2, 24.2),
                ylim=(-0.2, 12.2),
                aspect="equal",
                xlabel="x (m)" if row == 2 else "",
                ylabel="y (m)",
            )
            ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], color=color, lw=2, label=phase) for phase, color in colors.items()]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.047), ncol=4, frameon=False
    )
    fig.suptitle("Full missions: where the drone actually flew", fontsize=18, y=0.99)
    fig.text(
        0.04,
        0.025,
        "Top two rows use matched seeds. The scoped hidden mission uses a new seed. Squares mark final positions; dots mark launch.",
        fontsize=10,
    )
    fig.text(
        0.04,
        0.010,
        "Native XY traces over evaluator-only furniture. Jev receives architectural room data and sensed obstacles, not this hidden furniture map.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.01, 0.075, 0.99, 0.965), h_pad=3)
    output = Path("report/px4-clarity")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "flight-comparison.png", dpi=170)
    plt.close(fig)
    print(output / "flight-comparison.png")


if __name__ == "__main__":
    main()
