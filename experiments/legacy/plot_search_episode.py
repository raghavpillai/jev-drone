"""Render a saved search trajectory; never calls a model or changes an episode."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def faces(lo, hi):
    x, y, z = lo
    xx, yy, zz = hi
    a, b, c, d = (x, y, z), (xx, y, z), (xx, yy, z), (x, yy, z)
    e, f, g, h = (x, y, zz), (xx, y, zz), (xx, yy, zz), (x, yy, zz)
    return [(a, b, c, d), (e, f, g, h), (a, b, f, e), (b, c, g, f), (c, d, h, g), (d, a, e, h)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.episode.read_text())
    trace = np.asarray(data["trace"])
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    space = fig.add_subplot(grid[0, 0], projection="3d")
    plan = fig.add_subplot(grid[0, 1])
    alt = fig.add_subplot(grid[1, :])
    fig.patch.set_facecolor("#f5f2ea")
    for obj in data["ground_truth"]["objects"]:
        lo, hi = obj["box"]["lo"], obj["box"]["hi"]
        target = obj["id"] in data["eligible_target_ids"]
        color = "#d7782b" if target else "#b8b9af"
        space.add_collection3d(
            Poly3DCollection(
                faces(lo, hi), facecolor=color, edgecolor="#888c83", alpha=0.32, linewidth=0.4
            )
        )
        plan.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], facecolor=color, alpha=0.5))
        if target:
            plan.text(lo[0], hi[1] + 0.15, "target", fontsize=8, color="#93461f")
    points = trace[:, 1:3]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap="viridis", norm=plt.Normalize(0, 4), linewidth=1.8)
    line.set_array(trace[:-1, 3])
    plan.add_collection(line)
    space.plot(trace[:, 1], trace[:, 2], trace[:, 3], color="#23776b", linewidth=1.4)
    for ax in (plan, space):
        ax.set(xlim=(0, 18), ylim=(0, 10), xlabel="x / m", ylabel="y / m")
    space.set(zlim=(0, 4), zlabel="height / m", title="Actual 3D trajectory")
    space.set_box_aspect((18, 10, 6))
    space.view_init(25, -60)
    plan.set(aspect="equal", title="Top view · path color shows height")
    for wall in data["world"]["obstacles"][:4]:
        lo, hi = wall["lo"], wall["hi"]
        plan.add_patch(
            Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], facecolor="#737d70", alpha=0.5)
        )
        space.add_collection3d(
            Poly3DCollection(faces(lo, hi), facecolor="#737d70", alpha=0.06, linewidth=0.2)
        )
    for x, name in ((3, "living room"), (9, "bedroom"), (15, "study")):
        plan.text(x, 9.7, name, ha="center", fontsize=8)
    plan.scatter(*points[0], color="#226d5f", s=35, zorder=4)
    plan.scatter(*points[-1], color="#b93f3c", marker="x", s=50, zorder=4)
    fig.colorbar(line, ax=plan, label="Height / m", shrink=0.7)
    alt.plot(trace[:, 0], trace[:, 3], color="#23776b")
    for task in data["tasks"]:
        if task["outcome"] == "viewpoint_inspected":
            alt.scatter(
                task["time"], task["position"][2], marker="o", color="#7a5086", s=25, zorder=3
            )
    alt.set(
        xlim=(0, data["simulation_seconds"]),
        ylim=(0, 4),
        xlabel="Simulation time / seconds",
        ylabel="Height / m",
        title="Purple dots: completed stop-and-inspect viewpoints",
    )
    fig.suptitle(
        f"{data['item']} in {data['room'].replace('_', ' ')} · {data['status'].replace('_', ' ')} · {data['simulation_seconds']:.1f}s\n"
        f"{data['trial_id']} | All decisions: Jev through OpenRouter | Boxes shown only for evaluation",
        fontsize=13,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
