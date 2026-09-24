"""Run reproducible episodes; model latency advances physics under the old command."""
from jev_drone.evidence import snapshot_sources

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import itertools

from experiments.legacy.policy import CRITERIA, INSTRUCTIONS, INTENT_CRITERIA, INTENT_INSTRUCTIONS, PROMPT_VERSION, Jev, baseline, load_key
from experiments.legacy.sim import Simulation, scenario


def sample_history(history, limit=24):
    if len(history) <= limit:
        return list(history)
    return [history[round(i*(len(history)-1)/(limit-1))] for i in range(limit)]


def episode(args, name, seed, controller):
    sim = Simulation(scenario(name, seed), speed=args.speed)
    history, decisions = [], []
    rng = random.Random(seed)
    cost = 0.
    intent = "direct"
    while sim.status == "running" and sim.time < args.seconds-1e-9 and len(decisions) < args.max_calls:
        observation = sim.observation(history)
        observation["persistent_intent"] = intent
        if getattr(args, "dense_depth", False):
            rays = {}
            for x, y, z in itertools.product([-1, 0, 1], repeat=3):
                if not z or (not x and not y):
                    continue
                parts = []
                for value, positive, negative in zip((x,y,z), ("forward","left","up"), ("back","right","down")):
                    if value:
                        parts.append(positive if value > 0 else negative)
                rays["_".join(parts)] = round(sim.clearance((x,y,z)), 2)
            observation["depth_rays"] = rays
        if getattr(args, "long_memory", False):
            # Uniformly subsample the full observation history; no map, route,
            # reward, or recommended action is computed here.
            observation["breadcrumbs"] = sample_history(history)
        sent = sim.time
        if controller:
            decision = controller.decide(observation)
        else:
            decision = {"action": baseline(observation, args.policy, rng), "latency_seconds": args.baseline_delay}
        latency = decision["latency_seconds"]
        delay = latency if args.timing == "delayed" else 0.
        sim.advance(min(delay, max(0., args.seconds-sim.time)))
        received = sim.time
        accepted = sim.status == "running" and not decision.get("error") and delay <= args.max_age and sim.time < args.seconds
        if sim.status == "running":
            sim.command(decision["action"] if accepted else "brake")
        if accepted:
            intent = decision.get("intent", intent)
        decisions.append({"sent_at": round(sent, 4), "received_at": round(received, 4),
                          "accepted": accepted, "observation": observation, **decision})
        history.append({"position": observation["position_metres_xyz"], "goal_direction": observation["goal"]["direction"],
                        "goal_distance": observation["goal"]["distance_metres"],
                        "action": decision["action"] if accepted else "brake"})
        cost += decision.get("usage", {}).get("cost", 0.) or 0.
        # A maximum 5 Hz decision rate. Lockstep is explicitly diagnostic: wait
        # for inference without motion, then apply a 200 ms action.
        padding = max(0., .2-delay) if args.timing == "delayed" else .2
        sim.advance(min(padding, max(0., args.seconds-sim.time)))
        if cost >= args.episode_budget:
            if sim.status == "running":
                sim.status = "budget_limit"
            break
    if sim.status == "running":
        sim.status = "timeout" if sim.time >= args.seconds-1e-9 else "call_limit"
    result = sim.summary()
    result.update({"policy": args.policy, "timing": args.timing, "decisions": decisions,
                   "calls": len(decisions) if controller else 0,
                   "cost_usd": cost, "prompt_version": PROMPT_VERSION,
                   "errors": sum(bool(d.get("error")) for d in decisions),
                   "stale_decisions": sum(d["latency_seconds"] > args.max_age for d in decisions) if args.timing == "delayed" else 0,
                   "max_age_seconds": args.max_age})
    result["variant"] = {"dense_depth": getattr(args,"dense_depth",False), "long_memory": getattr(args,"long_memory",False)}
    result["variant"]["simple_controls"] = getattr(args,"simple_controls",False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["jev", "greedy", "reactive", "random"], default="jev")
    parser.add_argument("--timing", choices=["delayed", "lockstep"], default="delayed")
    parser.add_argument("--scenarios", nargs="+", choices=["open", "pillar", "wall", "overpass", "u_trap"], default=["open", "pillar", "wall", "overpass", "u_trap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--seconds", type=float, default=35.)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--max-calls", type=int, default=150)
    parser.add_argument("--max-age", type=float, default=.8)
    parser.add_argument("--baseline-delay", type=float, default=.3)
    parser.add_argument("--episode-budget", type=float, default=.025)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--dense-depth", action="store_true")
    parser.add_argument("--long-memory", action="store_true")
    parser.add_argument("--simple-controls", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.speed != 1.:
        parser.error("This prompt defines 1 m/s actions; change the prompt and experiment together before changing speed.")
    if args.out.exists():
        parser.error("Output already exists; use a new run directory to preserve evidence.")
    args.out.mkdir(parents=True)
    hashes = snapshot_sources(args.out / "source", experiments=True)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "key_file"},
                "source_sha256": hashes,
                "prompt": INSTRUCTIONS, "criteria": CRITERIA,
                "intent_prompt": INTENT_INSTRUCTIONS, "intent_criteria": INTENT_CRITERIA}
    (args.out/"metadata.json").write_text(json.dumps(metadata, indent=2))
    controller = Jev(load_key(args.key_file), args.dense_depth, args.long_memory, args.simple_controls) if args.policy == "jev" else None
    total_cost = 0.
    try:
        for seed in args.seeds:
            for name in args.scenarios:
                started = time.perf_counter()
                result = episode(args, name, seed, controller)
                result["wall_seconds"] = round(time.perf_counter()-started, 3)
                (args.out/f"{name}-{seed}.json").write_text(json.dumps(result, indent=2))
                total_cost += result["cost_usd"]
                print(json.dumps({"scenario": name, "seed": seed, "status": result["status"],
                                  "seconds": result["simulation_seconds"], "distance": result["final_distance_metres"],
                                  "calls": result["calls"], "cost_usd": round(result["cost_usd"], 6),
                                  "errors": result["errors"], "stale": result["stale_decisions"]}), flush=True)
    finally:
        if controller:
            controller.close()
    print(f"Total cost: ${total_cost:.6f}", flush=True)


if __name__ == "__main__":
    main()
