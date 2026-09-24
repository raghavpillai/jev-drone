"""Live partially observed object search: Jev perception judgments and local movement."""
from jev_drone.evidence import snapshot_sources
import argparse
import itertools
import json
from pathlib import Path
import time

from experiments.legacy.hierarchy import ask
from experiments.legacy.policy import Jev, load_key
from experiments.legacy.search_world import PORTALS, ROOMS, evaluate_report, ground_truth, make_home, room_at, search_waypoints, update_memory, visible_objects
from experiments.legacy.sim import Simulation, norm, subtract

SEARCH_INSTRUCTIONS = """Choose the next step to find and approach the requested object.
The room layout and doors are known, but furniture locations were unknown at launch. Only observed_objects have been seen.
recognized_target is an earlier semantic identification, not proof of arrival. If identified, approach it, then report it only when currently visible, within 2.1 metres of its observed surface, and stopped.
If no target is identified, actively search new viewpoints. Use the mission's requested room if specified; otherwise choose promising rooms and cover areas you have not yet inspected.
To enter a different room, first use the doorway approach in the current room, then the approach on the other side. Use connectivity; walls cannot be crossed elsewhere.
Persist with an unfinished waypoint unless stalled or the target is discovered. Do not repeat completed search viewpoints while unvisited ones remain. A visited point does NOT prove its entire room empty. Higher viewpoints can reveal occluded furniture.
Remember attempts and completed viewpoints to avoid loops. After a stalled approach choose a different viewpoint or approach.
Only report_not_found after thorough searching of the requested room or, for an unspecified room, all three rooms. Never declare absence just because the current view has no target.
You choose every route and search step. The local controller handles nearby obstacles; no route search or automatic exploration is performed for you."""


def planner_state(mission, sim, memory, visible, visited, attempts, target_name, report, recognized):
    points = search_waypoints()
    for object_id, item in memory.items():
        points[f"approach_{object_id}"] = {"xyz": item["approach_xyz"], "room": item["room"],
                                            "description": f"Approach observed {object_id}: {item['appearance']}"}
    candidates = {name: {**p, "distance_metres": round(norm(subtract(p["xyz"], sim.position)), 2),
                         "completed_visits": visited.count(name), "attempts": attempts.count(name)}
                  for name, p in points.items()}
    state = {"mission": mission, "position_xyz": sim.position, "current_room": room_at(sim.position),
             "stopped": norm(sim.velocity) <= .15, "known_rooms": ROOMS, "known_doorways": PORTALS,
             "furniture_map": "Unknown except for objects actually observed below.",
             "sensor": "Ideal panoramic 3.5 m object detections with solid occlusion. No camera pixels.",
             "visible_now": visible, "observed_objects": list(memory.values()),
             "recognized_target": recognized, "candidate_waypoints": candidates,
             "completed_waypoints": visited, "previous_target": target_name, "local_report": report}
    criteria = {name: p["description"] for name, p in points.items()}
    for object_id in memory:
        criteria[f"report_found_{object_id}"] = f"Finish: {object_id} matches the mission, is currently visible within 2.1 m, and the drone is stopped."
    criteria["report_not_found"] = "Finish: relevant rooms were thoroughly searched and the requested object was not found."
    return state, criteria, points


def deepseek_ask(client, state, instructions, criteria, max_tokens=1024, reasoning_effort="low"):
    started = time.perf_counter()
    response = client.post("https://openrouter.ai/api/v1/chat/completions", timeout=30., json={
        "model": "deepseek/deepseek-v4.1-flash", "max_tokens": max_tokens,
        "reasoning": {"effort": reasoning_effort, "exclude": True},
        "messages": [{"role": "system", "content": instructions},
                     {"role": "user", "content": json.dumps({"state": state, "options": criteria})}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "next_step", "strict": True,
            "schema": {"type": "object", "properties": {"choice": {"type": "string", "enum": list(criteria)}},
                       "required": ["choice"], "additionalProperties": False}}}})
    response.raise_for_status()
    result = response.json()
    decision = {"latency_seconds": time.perf_counter()-started, "usage": result.get("usage", {}),
                "model": result.get("model"), "state": state}
    try:
        answer = result["choices"][0]
        decision["finish_reason"] = answer.get("finish_reason")
        content = answer["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            return {**decision, "error": "missing_structured_content"}
        choice = json.loads(content)["choice"]
        if choice not in criteria:
            return {**decision, "error": "unknown_planner_choice"}
    except (KeyError, IndexError, TypeError, ValueError):
        return {**decision, "error": "invalid_structured_response"}
    return {**decision, "choice": choice}


def depth_observation(sim, history, target, intent):
    observation = sim.observation(history, target=target)
    observation["persistent_intent"] = intent
    rays = {}
    for x, y, z in itertools.product([-1, 0, 1], repeat=3):
        if not z or not (x or y):
            continue
        parts = [positive if v > 0 else negative for v, positive, negative in
                 zip((x,y,z), ("forward", "left", "up"), ("back", "right", "down")) if v]
        rays["_".join(parts)] = round(sim.clearance((x,y,z)), 2)
    observation["depth_rays"] = rays
    return observation


def episode(args, local):
    home = make_home(args.seed, args.target_room, args.absent)
    sim = Simulation(home.world)
    memory, visits, attempts, plans, decisions, history = {}, [], [], [], [], []
    target, target_name, recognized, intent = None, None, "none", "direct"
    best, last_progress, last_plan = float("inf"), 0., 0.
    error = None

    def cost():
        return sum(p.get("usage", {}).get("cost", 0) or 0 for p in plans+decisions)

    def global_call(kind, state, instructions, criteria):
        sim.command("brake")
        sent = sim.time
        started = time.perf_counter()
        fn = deepseek_ask if kind == "plan" and args.planner == "deepseek" else ask
        try:
            decision = fn(local.client, state, instructions, criteria)
        except Exception as exc:
            decision = {"error": type(exc).__name__, "latency_seconds": time.perf_counter()-started, "state": state}
        plans.append({"kind": kind, "sent_at": sent, **decision})
        sim.advance(min(decision["latency_seconds"], max(0., args.seconds-sim.time)))
        if decision.get("error"):
            raise RuntimeError(decision["error"])
        return decision["choice"]

    try:
        while sim.status == "running" and sim.time < args.seconds-1e-9 and len(decisions) < 450 and len(plans) < 80:
            if cost() >= args.budget:
                sim.status = "budget_limit"
                break
            visible = visible_objects(home, sim.position)
            new_ids = {o["id"] for o in visible} - memory.keys()
            update_memory(memory, visible, sim.position, sim.time)
            if new_ids:
                recognized = global_call("recognize", {"mission": args.mission,
                    "observed_objects": [{"id": k, "appearance": v["appearance"], "room": v["room"]} for k, v in memory.items()]},
                    "Which observed object matches the object requested in the mission? Use its physical appearance and any specified room. Select none if there is no matching observation. Closed cupboards, plants, and tables are not bookcases.",
                    {"none": "No observed object matches the request.", **{k: v["appearance"] for k, v in memory.items()}})
                if sim.status != "running" or sim.time >= args.seconds-1e-9:
                    break
                if cost() >= args.budget:
                    sim.status = "budget_limit"
                    break
            arrived = target is not None and norm(subtract(target, sim.position)) <= .65 and norm(sim.velocity) <= .15
            stalled = target is not None and sim.time-last_progress > 7.
            if target is None or arrived or stalled or new_ids or sim.time-last_plan > 10.:
                if arrived and target_name not in visits:
                    visits.append(target_name)
                report = "arrived" if arrived else "stalled" if stalled else "new objects observed" if new_ids else "progress update"
                visible = visible_objects(home, sim.position)
                state, criteria, points = planner_state(args.mission, sim, memory, visible, visits, attempts, target_name, report, recognized)
                choice = global_call("plan", state, SEARCH_INSTRUCTIONS, criteria)
                print(json.dumps({"seed": args.seed, "planner": args.planner, "time": round(sim.time, 2),
                                  "choice": choice, "recognized": recognized, "seen": len(memory)}), flush=True)
                if sim.status != "running" or sim.time >= args.seconds-1e-9:
                    break
                if choice.startswith("report_"):
                    sim.status = evaluate_report(choice, home, sim, visible_objects(home, sim.position))
                    break
                if choice != target_name:
                    history, intent = [], "direct"
                target_name, target = choice, tuple(points[choice]["xyz"])
                attempts.append(choice)
                best, last_progress, last_plan = norm(subtract(target, sim.position)), sim.time, sim.time
            if sim.status != "running" or sim.time >= args.seconds-1e-9:
                break
            observation = depth_observation(sim, history, target, intent)
            sent = sim.time
            decision = local.decide(observation)
            latency = decision["latency_seconds"]
            sim.advance(min(latency, max(0., args.seconds-sim.time)))
            accepted = sim.status == "running" and sim.time < args.seconds-1e-9 and latency <= .8 and not decision.get("error")
            if sim.status == "running":
                sim.command(decision["action"] if accepted else "brake")
            if accepted:
                intent = decision["intent"]
            decisions.append({"sent_at": sent, "received_at": sim.time, "accepted": accepted,
                              "target": target_name, "observation": observation, **decision})
            history.append({"position": observation["position_metres_xyz"], "goal_direction": observation["goal"]["direction"],
                            "goal_distance": observation["goal"]["distance_metres"], "action": decision["action"] if accepted else "brake"})
            sim.advance(min(max(0., .2-latency), max(0., args.seconds-sim.time)))
            distance = norm(subtract(target, sim.position))
            if distance < best-.15:
                best, last_progress = distance, sim.time
        if sim.status == "running":
            sim.status = "timeout" if sim.time >= args.seconds-1e-9 else "call_limit"
    except Exception as exc:
        error = type(exc).__name__
        sim.status = "planner_or_runtime_error"
    sim.record()
    result = sim.summary()
    result.pop("final_distance_metres")  # Sentinel goal is not a search metric.
    result.update({"mission": args.mission, "planner": args.planner, "ground_truth": ground_truth(home),
                   "plans": plans, "decisions": decisions, "observed_objects": memory, "visited_waypoints": visits,
                   "recognized_target": recognized, "target_seen": home.target_id in memory,
                   "target_identified": home.target_id is not None and recognized == home.target_id,
                   "calls": len(plans)+len(decisions), "cost_usd": cost(), "error": error,
                   "local_accepted_hz": sum(d["accepted"] for d in decisions)/max(sim.time, .001),
                   "target_first_seen_seconds": memory.get(home.target_id, {}).get("first_seen_at")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-room", choices=list(ROOMS), default="bedroom", help="Evaluator-only hidden placement")
    parser.add_argument("--absent", action="store_true", help="Evaluator-only target absence")
    parser.add_argument("--mission", default="Find a bookcase somewhere in this home. Approach it, stop nearby, and report it found.")
    parser.add_argument("--planner", choices=["jev", "deepseek"], default="jev")
    parser.add_argument("--seconds", type=float, default=150.)
    parser.add_argument("--budget", type=float, default=.08)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    args.out.mkdir(parents=True)
    snapshot_sources(args.out / "source", experiments=True)
    (args.out/"metadata.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items() if k != "key_file"}, indent=2))
    local = Jev(load_key(args.key_file), dense_depth=True, simple_controls=True)
    try:
        result = episode(args, local)
    finally:
        local.close()
    (args.out/"episode.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["status", "simulation_seconds", "calls", "target_seen", "target_identified", "cost_usd", "error"]}), flush=True)


if __name__ == "__main__":
    main()
