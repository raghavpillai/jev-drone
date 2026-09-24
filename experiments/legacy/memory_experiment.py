"""Matched local-memory/recovery experiments; Jev owns all decisions."""

import argparse
import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.legacy.jev_navigation import Navigation
from experiments.legacy.representation_experiment import world
from experiments.legacy.sim import Simulation
from jev_drone.evidence import snapshot_sources
from jev_drone.gateway import JevGateway, load_credential


def run(provider, key, mode, case, seed, seconds, limited, degraded):
    sim = Simulation(world(case, seed))
    gateway = JevGateway(provider, key)
    nav = Navigation(gateway, mode, limited, degraded, seed)
    calls = []
    try:
        while sim.status == "running" and sim.time < seconds - 1e-9 and len(calls) < 350:
            if nav.target is not None:
                recovery = nav.recovery(sim, sim.world.goal, {"status": "no recent progress"})
                if recovery:
                    sim.command("brake")
                    calls.append({"role": "recovery", "sent_at": sim.time, **recovery})
                    sim.advance(min(recovery["latency_seconds"], max(0.0, seconds - sim.time)))
            if sim.status != "running" or sim.time >= seconds - 1e-9:
                break
            sent = sim.time
            decision = nav.decide(sim, sim.world.goal)
            latency = decision["latency_seconds"]
            sim.advance(min(latency, max(0.0, seconds - sim.time)))
            accepted = (
                sim.status == "running"
                and sim.time < seconds - 1e-9
                and latency <= 0.8
                and not decision.get("error")
            )
            if sim.status == "running":
                sim.command(decision["action"] if accepted else "brake")
            nav.accept(sim, decision, accepted, sim.world.goal)
            calls.append({"role": "control", "sent_at": sent, "accepted": accepted, **decision})
            sim.advance(min(max(0.0, 0.2 - latency), max(0.0, seconds - sim.time)))
    finally:
        gateway.close()
    if sim.status == "running":
        sim.status = "timeout" if sim.time >= seconds - 1e-9 else "call_limit"
    sim.record()
    return {
        **sim.summary(),
        "memory_mode": mode,
        "case": case,
        "seed": seed,
        "provider": provider,
        "limited_view": limited,
        "degraded_depth": degraded,
        "calls": calls,
        "recovery_events": nav.events,
        "observed_map_cells": len(nav.map.cells),
        "cost_usd": sum(c.get("usage", {}).get("cost", 0) or 0 for c in calls),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openrouter", "vercel"], default="openrouter")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["history", "map", "recovery"],
        default=["history", "map", "recovery"],
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["doorway", "descent", "overpass", "u_trap"],
        default=["overpass", "u_trap"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seconds", type=float, default=75.0)
    parser.add_argument("--limited", action="store_true")
    parser.add_argument("--degraded", action="store_true")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = load_credential(args.provider, args.key_file)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    (args.out / "metadata.json").write_text(
        json.dumps(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
                if k != "key_file"
            },
            indent=2,
        )
    )
    jobs = list(itertools.product(args.modes, args.cases, args.seeds))
    random.Random(922).shuffle(jobs)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(run, args.provider, key, m, c, s, args.seconds, args.limited, args.degraded)
            for m, c, s in jobs
        ]
        for future in as_completed(futures):
            d = future.result()
            (args.out / f"{d['memory_mode']}-{d['case']}-{d['seed']}.json").write_text(
                json.dumps(d, indent=2)
            )
            print(
                json.dumps(
                    {
                        k: d[k]
                        for k in [
                            "memory_mode",
                            "case",
                            "seed",
                            "status",
                            "simulation_seconds",
                            "cost_usd",
                        ]
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
