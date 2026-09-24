"""Plot recorded native setpoints and motion from the scripted patch check."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    epoch = result["world"]["mission_start_sim_time"]
    wire = [s for s in result["setpoints"] if s["sim"] >= epoch]
    trace = result["trace"]
    release = next(c["time"] for c in result["commands"] if c.get("patch") == {"x": 0.0})
    stop_up = next(c["time"] for c in result["commands"] if c.get("patch") == {"z": 0.0})
    expiry = next(
        e["sim"] - epoch
        for e in result["command_expirations"]
        if e["sim"] >= epoch and any(e["before"]["velocity_mps"].values())
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, constrained_layout=True)
    for ax, component, label, color in zip(
        axes, (0, 2), ("Forward / east", "Up"), ("#2869a6", "#16805d")
    ):
        ax.step(
            [s["sim"] - epoch for s in wire],
            [s["velocity"][component] for s in wire],
            where="post",
            label="Published control",
            color=color,
            linewidth=2,
        )
        ax.plot(
            [s["time"] for s in trace],
            [s["velocity"][component] for s in trace],
            label="Actual simulated velocity",
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax.axhline(0, color="#777", linewidth=0.6)
        ax.axvline(release, color="#aaa", linewidth=1)
        ax.axvline(expiry, color="#ba4b38", linewidth=1, linestyle=":")
        ax.set(ylabel=label + " (m/s)", ylim=(-0.08, 0.37))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    axes[0].annotate(
        'Update {"x": 0}',
        (release, 0.25),
        xytext=(release + 0.25, 0.32),
        arrowprops={"arrowstyle": "->", "color": "#555"},
        fontsize=10,
    )
    axes[1].annotate(
        "Upward request remains active",
        ((release + stop_up) / 2, 0.25),
        xytext=(2.9, 0.32),
        arrowprops={"arrowstyle": "->", "color": "#555"},
        fontsize=10,
    )
    axes[0].annotate(
        "No heartbeat: lease expires\nStale patch rejected",
        (expiry, 0.03),
        xytext=(expiry + 0.25, 0.22),
        arrowprops={"arrowstyle": "->", "color": "#ba4b38"},
        fontsize=10,
    )
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_xlabel("Seconds after scripted native control check begins")
    fig.suptitle(
        "Partial updates preserve untouched controls; silence still stops the drone", fontsize=13
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    fig.savefig(args.out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
