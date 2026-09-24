"""Jev global waypoint selection plus Jev local seven-command movement."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from experiments.legacy.house import DESTINATIONS, PORTALS, ROOMS, WAYPOINTS, locate_room, world
from experiments.legacy.policy import ENDPOINT, MODEL, Jev, load_key
from experiments.legacy.sim import Simulation, norm, subtract
from jev_drone.evidence import snapshot_sources


def ask(client, state, instructions, criteria):
    started = time.perf_counter()
    response = client.post(
        ENDPOINT,
        json={
            "model": MODEL,
            "state": state,
            "questions": {
                "selection": {"type": "choice", "instructions": instructions, "criteria": criteria}
            },
        },
    )
    response.raise_for_status()
    data = response.json()
    answer = data["answers"]["selection"]
    if answer["choice"] not in criteria:
        raise ValueError("Unknown planner choice")
    return {
        "choice": answer["choice"],
        "answer": answer,
        "latency_seconds": time.perf_counter() - started,
        "usage": data.get("usage", {}),
        "model": data.get("model"),
        "state": state,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", required=True)
    parser.add_argument(
        "--expected",
        choices=list(DESTINATIONS),
        required=True,
        help="Evaluator ground truth; never passed as the parsed model answer",
    )
    parser.add_argument("--start-room", choices=list(ROOMS), default="living_room")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Bypass global waypoint selection; still resolve the mission with Jev",
    )
    parser.add_argument("--seconds", type=float, default=55.0)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    args.out.mkdir(parents=True)
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
    local = Jev(load_key(args.key_file), simple_controls=True)
    sim = Simulation(world(args.expected, args.start_room))
    plans, decisions, visited, history = [], [], [], []
    intent, target_name, target = "direct", None, None
    best_distance, last_progress = float("inf"), 0.0
    error = None
    try:
        resolution = ask(
            local.client,
            {"mission": args.mission, "destinations": DESTINATIONS},
            "Which named destination did the user request? Select the matching room or object.",
            {k: v["description"] for k, v in DESTINATIONS.items()},
        )
        plans.append({"kind": "resolve", "sent_at": sim.time, **resolution})
        sim.advance(resolution["latency_seconds"])
        destination = resolution["choice"]
        if destination != args.expected:
            sim.status = "wrong_destination"
        candidates = {**WAYPOINTS, "destination": DESTINATIONS[destination]}
        while (
            sim.status == "running"
            and sim.time < args.seconds - 1e-9
            and len(decisions) < 180
            and len(plans) < 20
        ):
            arrived = (
                target is not None
                and target_name != "destination"
                and norm(subtract(target, sim.position)) < 0.65
                and norm(sim.velocity) < 0.15
            )
            stalled = target is not None and sim.time - last_progress > 8.0
            if target is None or arrived or stalled:
                if arrived:
                    visited.append(target_name)
                if args.flat:
                    target_name = "destination"
                else:
                    sim.command("brake")
                    state = {
                        "mission": args.mission,
                        "destination": DESTINATIONS[destination],
                        "current_room": locate_room(sim.position),
                        "position_xyz": sim.position,
                        "rooms": ROOMS,
                        "portals": PORTALS,
                        "obstacles": [asdict(b) for b in sim.world.obstacles],
                        "waypoints": candidates,
                        "reached_waypoints": visited,
                        "previous_target": target_name,
                        "local_report": "stalled"
                        if stalled
                        else "arrived"
                        if arrived
                        else "starting",
                    }
                    plan = ask(
                        local.client,
                        state,
                        "Select the next intermediate waypoint for this mission. Rooms are separated by solid walls except at the connecting doorway. "
                        "Cross through the doorway by first reaching its approach on your current room's side, then the opposite side. "
                        "Once in the destination's room, choose destination. Do not repeat an already reached waypoint unless recovering from a failed approach. "
                        "The local controller will avoid furniture; do not send it directly through a wall. Use room coordinates, portal connectivity and reached waypoints.",
                        {k: v["description"] for k, v in candidates.items()},
                    )
                    plans.append({"kind": "waypoint", "sent_at": sim.time, **plan})
                    sim.advance(min(plan["latency_seconds"], args.seconds - sim.time))
                    target_name = plan["choice"]
                target = tuple(candidates[target_name]["xyz"])
                history = []
                best_distance = norm(subtract(target, sim.position))
                last_progress = sim.time
                intent = "direct"
                print(
                    json.dumps(
                        {
                            "next_waypoint": target_name,
                            "xyz": target,
                            "sim_seconds": round(sim.time, 2),
                        }
                    ),
                    flush=True,
                )
            if sim.status != "running" or sim.time >= args.seconds:
                break
            observation = sim.observation(history, target=target)
            observation["persistent_intent"] = intent
            sent = sim.time
            decision = local.decide(observation)
            latency = decision["latency_seconds"]
            sim.advance(min(latency, args.seconds - sim.time))
            accepted = (
                sim.status == "running"
                and sim.time < args.seconds
                and latency <= 0.8
                and not decision.get("error")
            )
            if sim.status == "running":
                sim.command(decision["action"] if accepted else "brake")
            if accepted:
                intent = decision["intent"]
            decisions.append(
                {
                    "sent_at": sent,
                    "received_at": sim.time,
                    "accepted": accepted,
                    "target": target_name,
                    "observation": observation,
                    **decision,
                }
            )
            history.append(
                {
                    "position": observation["position_metres_xyz"],
                    "goal_direction": observation["goal"]["direction"],
                    "goal_distance": observation["goal"]["distance_metres"],
                    "action": decision["action"] if accepted else "brake",
                }
            )
            sim.advance(min(max(0.0, 0.2 - latency), max(0.0, args.seconds - sim.time)))
            distance = norm(subtract(target, sim.position))
            if distance < best_distance - 0.15:
                best_distance, last_progress = distance, sim.time
            cost = sum(d.get("usage", {}).get("cost", 0) or 0 for d in decisions + plans)
            if cost >= 0.05 and sim.status == "running":
                sim.status = "budget_limit"
        if sim.status == "running":
            sim.status = "timeout" if sim.time >= args.seconds - 1e-9 else "call_limit"
    except Exception as exc:
        error = type(exc).__name__
        sim.status = "planner_or_runtime_error"
    finally:
        local.close()
        result = sim.summary()
        result.update(
            {
                "mission": args.mission,
                "expected": args.expected,
                "plans": plans,
                "decisions": decisions,
                "policy": "jev-flat" if args.flat else "jev-hierarchical",
                "timing": "delayed",
                "calls": len(plans) + len(decisions),
                "errors": sum(bool(d.get("error")) for d in decisions) + bool(error),
                "stale_decisions": sum(d["latency_seconds"] > 0.8 for d in decisions),
                "cost_usd": sum(d.get("usage", {}).get("cost", 0) or 0 for d in decisions + plans),
                "error": error,
            }
        )
        (args.out / "episode.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ["status", "simulation_seconds", "calls", "cost_usd", "error"]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
