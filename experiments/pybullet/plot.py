"""Scientific 3D trajectory and altitude plots from a recorded physical episode."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def faces(center, half):
    lo, hi = np.array(center)-half, np.array(center)+half
    vertices = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return [vertices[index] for index in ([0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                                          [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.episode.read_text())
    rooms = result['world']['rooms'].values()
    lower = np.min([bounds[0] for bounds in rooms],axis=0)
    upper = np.max([bounds[1] for bounds in rooms],axis=0)
    plt.style.use("dark_background")
    figure = plt.figure(figsize=(14, 7), facecolor="#10171f")
    grid = figure.add_gridspec(2, 3, width_ratios=(1, 1, 1.1))
    ax = figure.add_subplot(grid[:, :2], projection="3d", facecolor="#10171f")
    for box in result["world"]["geometry"]:
        if box["name"] in ("floor", "ceiling", "north_wall", "south_wall", "east_wall", "west_wall", "lintel"):
            continue
        opacity = .10 if box["name"] == "wall" else .28
        ax.add_collection3d(Poly3DCollection(faces(box["center"], np.array(box["half"])),
                                            facecolors=[box["color"][:3]], edgecolors="#8495a3", linewidths=.3, alpha=opacity))
    position = np.array([t["position"] for t in result["trace"]])
    times = np.array([t["time"] for t in result["trace"]])
    ax.plot(*position.T, color="#52dce8", linewidth=2.3)
    ax.scatter(*position[0], color="#80ef9a", s=60, label="Launch")
    ax.scatter(*position[-1], color="#ffc96a", s=60, label="Finish")
    ax.set(xlim=(lower[0],upper[0]), ylim=(lower[1],upper[1]), zlim=(lower[2],upper[2]), xlabel="x (m)", ylabel="y (m)", zlabel="Altitude (m)")
    ax.set_box_aspect(upper-lower)
    ax.view_init(elev=27, azim=-57)
    ax.legend(loc="upper left", frameon=False)
    height = figure.add_subplot(grid[0, 2], facecolor="#10171f")
    height.plot(times, position[:, 2], color="#52dce8")
    height.set(xlabel="Simulated time (s)", ylabel="Altitude (m)", ylim=(lower[2],upper[2]))
    height.grid(alpha=.15)
    speed = figure.add_subplot(grid[1, 2], facecolor="#10171f")
    speed.plot(times, [np.linalg.norm(t["velocity"]) for t in result["trace"]], color="#ffc96a")
    speed.set(xlabel="Simulated time (s)", ylabel="Speed (m/s)")
    speed.grid(alpha=.15)
    for report in result["reports"]:
        height.axvline(report["time"], color="#80ef9a", alpha=.6, linestyle=":")
        height.text(report["time"], upper[2]-.15, report["choice"], rotation=90, va="top", fontsize=8)
    figure.suptitle(result["id"]+" — "+result["status"], fontsize=18)
    figure.text(.05, .025, "Evaluator geometry shown for inspection only. Jev received rendered-sensor measurements; every flight movement was chosen by Jev.", fontsize=10, color="#a7b8c9")
    figure.subplots_adjust(left=.02, right=.96, bottom=.12, top=.88, wspace=.3, hspace=.32)
    figure.savefig(args.out, dpi=160, facecolor=figure.get_facecolor())


if __name__ == "__main__":
    main()
