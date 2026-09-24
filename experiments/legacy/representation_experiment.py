"""Compare Jev state encodings without changing observations or action choices."""
from jev_drone.evidence import snapshot_sources
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import itertools
import json
from pathlib import Path
import random
import time

from experiments.legacy.control_experiment import course
from experiments.legacy.policy import ENDPOINT, MODEL, INSTRUCTIONS, INTENT_INSTRUCTIONS, INTENT_CRITERIA, Jev, load_key
from experiments.legacy.sim import Box, Simulation, scenario

DIRECTIONS = list(itertools.product((-1, 0, 1), repeat=3))
DIRECTIONS.remove((0, 0, 0))
VARIANTS = ("numeric", "facts", "grid", "brief")
CASES = ("doorway", "descent", "overpass", "u_trap")


def label(direction):
    return "_".join(p if v > 0 else n for v, p, n in
                    zip(direction, ("forward", "left", "up"), ("back", "right", "down")) if v)


def world(case, seed):
    base = scenario("u_trap", 0) if case == "u_trap" else course(case)
    if seed:
        # Held-out geometry: narrower doorway / different obstacle heights and a mirrored room.
        obstacles = base.obstacles
        if case == "doorway":
            obstacles = (Box((5.85, 0, 0), (6.15, 4.5, 4)),
                         Box((5.85, 6.5, 0), (6.15, 10, 4)))
        elif case == "overpass":
            obstacles = (Box((5, 0, 0), (6, 10, 2.6)),)
        elif case == "descent":
            obstacles = (Box((1.95, 1.2, 0), (4.2, 3.5, 1.0)),)
        size = base.size[0]
        mirror = lambda p: (size-p[0], p[1], p[2])
        base = replace(base, start=mirror(base.start), goal=mirror(base.goal),
                       obstacles=tuple(Box((size-b.hi[0], b.lo[1], b.lo[2]),
                                           (size-b.lo[0], b.hi[1], b.hi[2])) for b in obstacles))
    return replace(base, seed=seed)


def observe(sim, history, intent, degraded, seed, target=None):
    common = sim.observation(history, target=target)
    common.pop("ranges")
    common.pop("sensor_description")
    common["persistent_intent"] = intent
    # Delay is from an actual past pose, never a shifted timestamp on fresh ranges.
    cutoff = max(0., sim.time-(.3 if degraded else 0.))
    frame = max((row for row in sim.trace if row[0] <= cutoff), key=lambda row: row[0])
    capture_time = frame[0] if degraded else sim.time
    capture_position = tuple(frame[1:4]) if degraded else sim.position
    sensor = Simulation(sim.world, sensor_range=3.)
    sensor.position = capture_position
    rng = random.Random(seed*1000003 + int(capture_time*10))
    readings = {}
    for direction in DIRECTIONS:
        value = sensor.clearance(direction)
        if degraded:
            noise, missing = rng.gauss(0., .15), rng.random() < .15
            value = None if missing else min(3., max(0., round((value+noise)*4)/4))
        readings[label(direction)] = None if value is None else round(value, 2)
    common["sensor"] = {"range_cap_m": 3., "age_seconds": round(sim.time-capture_time, 3),
                        "capture_position": [round(p, 2) for p in capture_position],
                        "missing": "unknown, not free", "size_adjusted_clearance": True}
    return common, readings


def bucket(value):
    if value is None:
        return "UNKNOWN"
    return "BLOCKED" if value < 1.3 else "OPEN"


def encode(common, readings, variant):
    if variant == "numeric":
        return {**common, "directional_clearance_metres": readings}
    if variant == "facts":
        return {**common, "spatial_facts": [
            f"{name}: {bucket(value)}" + (f", clearance {value} m" if value is not None else "")
            for name, value in readings.items()]}
    if variant == "brief":
        primary = ("forward", "back", "left", "right", "up", "down")
        describe = lambda name: f"{name}: {bucket(readings[name])}" + (f", clearance {readings[name]} m" if readings[name] is not None else "")
        return ("Current task, motion, observation age and memory:\n" + json.dumps(common) +
                "\nCLEARANCES FOR THE SIX MOVEMENT DIRECTIONS:\n" + "\n".join(describe(n) for n in primary) +
                "\nSUPPLEMENTARY DIAGONAL DEPTH OBSERVATIONS:\n" + "\n".join(describe(n) for n in readings if n not in primary))
    if variant != "grid":
        raise ValueError(variant)
    layers = {}
    for z, name in ((1, "upward"), (0, "horizontal"), (-1, "downward")):
        layers[name] = [["SELF" if (x, y, z) == (0, 0, 0) else readings[label((x, y, z))]
                         for y in (1, 0, -1)] for x in (1, 0, -1)]
    return {**common, "directional_depth_grid": layers,
            "grid_legend": "Each cell is a ray clearance in metres, NOT an occupied voxel. Rows forward/level/back; columns left/center/right. Layer supplies up/level/down. null=unknown. Center of upward/downward layers is straight up/down."}


GUIDANCE = """
You have 26 local depth directions, not a full map. A clearance under 1.3 m is BLOCKED; otherwise OPEN up to the measured range only. Missing/null is UNKNOWN, not OPEN. Measurements can be stale; consider age and velocity. There is no route solver or movement override.
Look at upward diagonal measurements for room above a barrier; a same-height destination can require climbing first. Fixed heading: +x forward, +y left, +z up. Speed 1 m/s, acceleration/braking 2 m/s^2, up to 0.5 s to stop. Prior movement continues during requests.
"""


def decide(local, state):
    started = time.perf_counter()
    try:
        response = local.client.post(ENDPOINT, json={"model": MODEL, "state": state, "questions": {
            "movement": {"type": "choice", "instructions": INSTRUCTIONS+GUIDANCE, "criteria": local.criteria},
            "intent": {"type": "choice", "instructions": INTENT_INSTRUCTIONS+GUIDANCE, "criteria": INTENT_CRITERIA}}})
        response.raise_for_status()
        data = response.json()
        action, intent = data["answers"]["movement"]["choice"], data["answers"]["intent"]["choice"]
        if action not in local.criteria or intent not in INTENT_CRITERIA:
            raise ValueError("invalid choice")
        return {"action": action, "intent": intent, "answers": data["answers"],
                "model": data.get("model"), "usage": data.get("usage", {}),
                "latency_seconds": time.perf_counter()-started}
    except Exception as exc:
        return {"action": "brake", "error": type(exc).__name__, "latency_seconds": time.perf_counter()-started}


def run(key, variant, case, degraded, seed, seconds):
    sim = Simulation(world(case, seed))
    history, calls, intent = [], [], "direct"
    local = Jev(key, simple_controls=True)
    try:
        while sim.status == "running" and sim.time < seconds-1e-9 and len(calls) < 160:
            common, readings = observe(sim, history, intent, degraded, seed)
            state = encode(common, readings, variant)
            sent = sim.time
            decision = decide(local, state)
            latency = decision["latency_seconds"]
            sim.advance(min(latency, max(0., seconds-sim.time)))
            accepted = sim.status == "running" and sim.time < seconds-1e-9 and latency <= .8 and not decision.get("error")
            if sim.status == "running":
                sim.command(decision["action"] if accepted else "brake")
            if accepted:
                intent = decision["intent"]
            history.append({"position": [round(p, 2) for p in sim.position],
                            "action": decision["action"] if accepted else "brake"})
            calls.append({"sent_at": sent, "accepted": accepted, "state": state, **decision})
            sim.advance(min(max(0., .2-latency), max(0., seconds-sim.time)))
    finally:
        local.close()
    if sim.status == "running":
        sim.status = "timeout"
    sim.record()
    return {**sim.summary(), "variant": variant, "case": case, "degraded": degraded,
            "decisions": calls, "calls": len(calls), "seed": seed,
            "cost_usd": sum(c.get("usage", {}).get("cost", 0) or 0 for c in calls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:3]))
    parser.add_argument("--conditions", nargs="+", choices=("clean", "degraded"), default=["clean", "degraded"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=40.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    (args.out/"protocol.json").write_text(json.dumps({"variants": args.variants, "conditions": args.conditions,
        "seed": args.seed, "seconds": args.seconds, "workers": 3, "cases": CASES,
        "degradation": {"delay_s": .3, "dropout": .15, "noise_sigma_m": .15, "quantization_m": .25}}, indent=2))
    key = load_key(args.key_file)
    jobs = list(itertools.product(args.variants, CASES, args.conditions))
    random.Random(921).shuffle(jobs)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run, key, v, c, d == "degraded", args.seed, args.seconds): (v, c, d)
                   for v, c, d in jobs}
        for future in as_completed(futures):
            v, c, d = futures[future]
            result = future.result()
            (args.out/f"{v}-{c}-{d}.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({k: result[k] for k in ("variant", "case", "degraded", "status", "calls", "simulation_seconds", "cost_usd")}), flush=True)


if __name__ == "__main__":
    main()
