"""Matched waypoint-control ablation. Every variant queries Jev for every movement."""

import argparse
import json
import time
from pathlib import Path

from experiments.legacy.control import choose_control
from experiments.legacy.policy import Jev, load_key
from experiments.legacy.search import depth_observation
from experiments.legacy.sim import Box, Scenario, Simulation
from jev_drone.evidence import snapshot_sources


def course(name):
    if name == "descent":
        return Scenario(
            name, 0, (1.6, 6.0, 2.8), (1.6, 2.5, 1.4), (Box((1.95, 1.2, 0), (4.2, 3.5, 0.75)),)
        )
    if name == "overpass":
        return Scenario(
            name, 0, (2.0, 5.0, 1.4), (9.0, 5.0, 1.4), (Box((5.0, 0, 0), (6.0, 10.0, 2.4)),)
        )
    return Scenario(
        name,
        0,
        (2.0, 5.5, 1.4),
        (8.5, 5.5, 1.4),
        (Box((5.85, 0, 0), (6.15, 4.0, 4.0)), Box((5.85, 7.0, 0), (6.15, 10.0, 4.0))),
    )


def run(args, controller):
    sim = Simulation(course(args.course))
    history, calls = [], []
    intent = "direct"
    while sim.status == "running" and sim.time < args.seconds - 1e-9 and len(calls) < 160:
        sent = sim.time
        started = time.perf_counter()
        try:
            if args.variant == "original":
                observation = depth_observation(sim, history, sim.world.goal, intent)
                decision = controller.decide(observation)
                decision["state"] = observation
                decision["speed"] = 1.0
            else:
                decision = choose_control(
                    controller.client,
                    sim,
                    sim.world.goal,
                    history,
                    {"task": "reach_waypoint_and_stop"},
                    slow=args.variant == "slow",
                )
        except Exception as exc:
            decision = {
                "action": "brake",
                "speed": 0.0,
                "latency_seconds": time.perf_counter() - started,
                "error": type(exc).__name__,
            }
        latency = decision["latency_seconds"]
        sim.advance(min(latency, max(0.0, args.seconds - sim.time)))
        accepted = (
            sim.status == "running"
            and sim.time < args.seconds - 1e-9
            and latency <= 0.8
            and not decision.get("error")
        )
        if sim.status == "running":
            sim.command(
                decision["action"] if accepted else "brake", decision["speed"] if accepted else 0.0
            )
        if accepted:
            intent = decision.get("intent", intent)
        history.append(
            {
                "position": [round(x, 2) for x in sim.position],
                "action": decision["action"],
                "speed": decision["speed"],
            }
        )
        calls.append({"sent_at": sent, "accepted": accepted, **decision})
        sim.advance(min(max(0.0, 0.2 - latency), max(0.0, args.seconds - sim.time)))
    if sim.status == "running":
        sim.status = "timeout"
    sim.record()
    return {
        **sim.summary(),
        "variant": args.variant,
        "decisions": calls,
        "calls": len(calls),
        "cost_usd": sum(d.get("usage", {}).get("cost", 0) or 0 for d in calls),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", choices=["descent", "overpass", "doorway"], required=True)
    parser.add_argument("--variant", choices=["original", "compact", "slow"], required=True)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    local = Jev(load_key(args.key_file), dense_depth=True, simple_controls=True)
    try:
        result = run(args, local)
    finally:
        local.close()
    (args.out / "episode.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: result[k] for k in ["variant", "status", "simulation_seconds", "calls", "cost_usd"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
