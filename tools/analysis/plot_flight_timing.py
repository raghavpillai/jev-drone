"""Separate network deadlines from physical flight progress in recorded trials."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.episode.read_text())
    controls = [c for c in result["calls"] if c["role"] == "control"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for accepted, color, label in (
        (True, "#40799e", "Applied"),
        (False, "#b4583d", "Rejected / request error"),
    ):
        selected = [c for c in controls if bool(c.get("accepted")) == accepted]
        axes[0].scatter(
            [c["time"] for c in selected],
            [c["latency_seconds"] for c in selected],
            s=5,
            c=color,
            label=label,
            alpha=0.7,
        )
    lease = result["world"]["config"]["command_lease_seconds"]
    axes[0].axhline(
        lease, color="#a23930", linestyle="--", linewidth=1, label=f"{lease:.1f} s control deadline"
    )
    axes[0].set_ylabel("API response seconds")
    axes[0].legend(loc="upper right", fontsize=8)
    trace = result["trace"]
    times = [t["time"] for t in trace]
    axes[1].plot(
        times, [np.linalg.norm(t["velocity"]) for t in trace], color="#32916e", linewidth=0.8
    )
    axes[1].set(xlabel="Simulated seconds", ylabel="Native speed (m/s)")
    for report in result["reports"]:
        if report["valid"]:
            for ax in axes:
                ax.axvline(report["time"], color="#94629c", linewidth=1, alpha=0.8)
            axes[1].text(
                report["time"],
                axes[1].get_ylim()[1] * 0.8,
                report["choice"],
                rotation=90,
                ha="right",
                color="#70477b",
            )
    fig.suptitle(f"{args.episode.parent.parent.name} / {result['id']} — {result['status']}")
    fig.text(
        0.05,
        0.01,
        "Late choices are rejected; the command lease expires to zero controls. Rejected calls also include API errors and control-state changes.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "timing.png", dpi=160)


if __name__ == "__main__":
    main()
