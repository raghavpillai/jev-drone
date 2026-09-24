"""Offline metrics and an evaluator-view rendering of partially observed search."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from experiments.legacy.search_world import ROOMS, room_at


def box_faces(lo, hi):
    x, y, z = lo
    X, Y, Z = hi
    return [[(x,y,z),(X,y,z),(X,Y,z),(x,Y,z)], [(x,y,Z),(X,y,Z),(X,Y,Z),(x,Y,Z)],
            [(x,y,z),(X,y,z),(X,y,Z),(x,y,Z)], [(x,Y,z),(X,Y,z),(X,Y,Z),(x,Y,Z)],
            [(x,y,z),(x,Y,z),(x,Y,Z),(x,y,Z)], [(X,y,z),(X,Y,z),(X,Y,Z),(X,y,Z)]]


def metrics(path, e):
    local = e["decisions"]
    plans = [p for p in e["plans"] if p["kind"] == "plan"]
    latencies = sorted(d["latency_seconds"] for d in local)
    recognition = [p for p in e["plans"] if p["kind"] == "recognize" and not p.get("error")]
    def recognition_correct(p):
        seen_ids = {o["id"] for o in p["state"]["observed_objects"]}
        expected = e["ground_truth"]["target_id"]
        return p["choice"] == (expected if expected in seen_ids else "none")
    return {"run": path.parent.name, "planner": e["planner"], "status": e["status"],
            "seconds": e["simulation_seconds"], "target_seen": e["target_seen"],
            "target_identified": e["target_identified"], "first_seen_seconds": e["target_first_seen_seconds"],
            "rooms_entered": sorted({room_at(p[1:4]) for p in e["trace"]}),
            "search_views_completed": {room: sum(v.startswith(f"search_{room}_") for v in e["visited_waypoints"]) for room in ROOMS},
            "peak_altitude_m": max(p[3] for p in e["trace"]),
            "local_accepted_hz": e["local_accepted_hz"],
            "local_median_ms": statistics.median(latencies)*1000 if latencies else None,
            "planner_median_ms": statistics.median(p["latency_seconds"] for p in plans)*1000 if plans else None,
            "calls": e["calls"], "global_calls": len(e["plans"]),
            "recognition_correct": sum(recognition_correct(p) for p in recognition),
            "recognition_calls": len(recognition),
            "errors": sum(bool(p.get("error")) for p in e["plans"]+local),
            "cost_usd": e["cost_usd"]}


def plot_episode(ax, e, title):
    objects = e["ground_truth"]["objects"]
    for wall in e["world"]["obstacles"][:4]:
        ax.add_collection3d(Poly3DCollection(box_faces(wall["lo"], wall["hi"]),
                            facecolor="#6b7785", edgecolor="#6b7785", alpha=.10, linewidth=.4))
    for obj in objects:
        color = "#e5a326" if obj["id"] == e["ground_truth"]["target_id"] else "#92aec3"
        ax.add_collection3d(Poly3DCollection(box_faces(obj["box"]["lo"], obj["box"]["hi"]),
                            facecolor=color, edgecolor="#536475", alpha=.4, linewidth=.5))
    trace = e["trace"]
    ax.plot([p[1] for p in trace], [p[2] for p in trace], [p[3] for p in trace], color="#127d67", lw=2)
    ax.scatter(*trace[0][1:4], color="#162b39", s=35)
    if e["target_first_seen_seconds"] is not None:
        first = min(trace, key=lambda p: abs(p[0]-e["target_first_seen_seconds"]))
        ax.scatter(*first[1:4], color="#d1641c", s=90, marker="*")
        ax.text(first[1], first[2], first[3]+.45, "First target sighting", fontsize=8)
    for i, room in enumerate(ROOMS):
        ax.text(i*6+1, -.8, 0, room.replace("_", " "), fontsize=9)
    ax.set(xlim=(0,18), ylim=(0,10), zlim=(0,4), xlabel="x (m)", ylabel="y (m)", zlabel="height (m)")
    ax.set_box_aspect((18,10,5))
    ax.view_init(elev=37, azim=-64)
    ax.set_title(f"{title}\n{e['status']} · {e['simulation_seconds']:.1f}s", fontsize=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path)
    args = parser.parse_args()
    paths = sorted(Path("results").glob("search-*/episode.json"))
    summary = [metrics(p, json.loads(p.read_text())) for p in paths]
    Path("report/search-summary.json").write_text(json.dumps(summary, indent=2))
    runs = args.runs or [Path("results/search-calibration-jev"), Path("results/search-eval-jev-study")]
    fig = plt.figure(figsize=(15,6), layout="constrained")
    for i, run in enumerate(runs[:2], 1):
        e = json.loads((run/"episode.json").read_text())
        plot_episode(fig.add_subplot(1, 2, i, projection="3d"), e,
                     "Jev: known target room" if "calibration" in run.name else "Jev: unknown target room")
    fig.suptitle("Search with unknown furniture: actual executed 3D paths\nEvaluator view: gold = target; all furniture was hidden from the model until sensed", fontsize=14)
    fig.savefig("report/search-navigation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
