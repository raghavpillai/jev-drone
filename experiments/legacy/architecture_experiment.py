"""Matched waypoint vs persistent-task planning; actual control is always Jev."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from experiments.legacy.experiment_world import make_scene
from experiments.legacy.hierarchy import ask
from experiments.legacy.policy import MODEL, Jev, load_key
from experiments.legacy.search import deepseek_ask, depth_observation
from experiments.legacy.search_world import (
    evaluate_report,
    ground_truth,
    update_memory,
    visible_objects,
)
from experiments.legacy.sim import Simulation, norm, subtract
from experiments.legacy.task_planning import (
    NAV_INSTRUCTIONS,
    TASK_INSTRUCTIONS,
    WAYPOINT_INSTRUCTIONS,
    all_checked,
    planning_state,
    room_coverage,
    task_options,
    waypoint_options,
)
from jev_drone.evidence import snapshot_sources


class Experiment:
    def __init__(self, args, local):
        self.args, self.local = args, local
        self.scene = make_scene(args.seed, args.case)
        self.sim = Simulation(self.scene.home.world)
        self.memory, self.checked, self.attempts, self.calls, self.history = {}, set(), [], [], []
        self.recognized, self.intent = "none", "direct"
        self.task, self.task_name, self.target, self.target_name = None, None, None, None
        self.best, self.last_progress = float("inf"), 0.0
        self.report = {"status": "starting"}
        self.task_events = []
        self.error = None

    def active(self):
        return self.sim.status == "running" and self.sim.time < self.args.seconds - 1e-9

    def cost(self):
        return sum(c.get("usage", {}).get("cost", 0) or 0 for c in self.calls)

    def request(self, role, state, instructions, criteria):
        if not self.active():
            raise TimeoutError("Episode deadline")
        if self.cost() >= self.args.budget:
            self.sim.status = "budget_limit"
            raise TimeoutError("Episode budget")
        # A deterministic pause, not another model choosing a movement.
        self.sim.command("brake")
        sent, started = self.sim.time, time.perf_counter()
        try:
            if role == "planner" and self.args.planner == "deepseek":
                result = deepseek_ask(
                    self.local.client, state, instructions, criteria, max_tokens=4096
                )
            else:
                result = ask(self.local.client, state, instructions, criteria)
        except Exception as exc:
            result = {
                "error": type(exc).__name__,
                "latency_seconds": time.perf_counter() - started,
                "state": state,
            }
        self.calls.append({"role": role, "sent_at": sent, **result})
        self.sim.advance(
            min(result["latency_seconds"], max(0.0, self.args.seconds - self.sim.time))
        )
        if result.get("error"):
            self.sim.status = "api_error"
            self.error = result["error"]
            return None
        return result["choice"] if self.active() else None

    def visible(self):
        return visible_objects(self.scene.home, self.sim.position)

    def observe(self):
        visible = self.visible()
        new_ids = {o["id"] for o in visible} - self.memory.keys()
        update_memory(self.memory, visible, self.sim.position, self.sim.time)
        for name, point in self.scene.points.items():
            if norm(subtract(point["xyz"], self.sim.position)) <= 0.65:
                self.checked.add(name)
        changed = False
        if new_ids:
            state = {
                "mission": self.scene.mission,
                "objects": [
                    {"id": k, "appearance": v["appearance"], "room": v["room"]}
                    for k, v in self.memory.items()
                ],
            }
            match = self.request(
                "recognition",
                state,
                "Which observed object matches the requested object and any requested room? Choose none if no observed object matches. A closed cupboard, plant, table, or bed is not a bookcase.",
                {
                    "none": "No observed match",
                    **{k: v["appearance"] for k, v in self.memory.items()},
                },
            )
            if match is not None:
                changed = match != self.recognized and match != "none"
                self.recognized = match
        return changed

    def state(self):
        return planning_state(
            self.scene,
            self.sim,
            self.memory,
            self.visible(),
            self.checked,
            self.recognized,
            self.report,
        )

    def finish(self, choice):
        if choice.startswith("report_found_") or choice == "report_not_found":
            status = evaluate_report(choice, self.scene.home, self.sim, self.visible())
            if status == "correct_absence" and not all_checked(self.scene, self.checked):
                status = "unsupported_absence"
            self.sim.status = status
            return True
        return False

    def assign_waypoint(self, name, points):
        if name != self.target_name:
            self.history, self.intent = [], "direct"
        self.target_name, self.target = name, tuple(points[name]["xyz"])
        self.attempts.append(name)
        self.best = norm(subtract(self.target, self.sim.position))
        self.last_progress = self.sim.time

    def choose_task(self):
        state = self.state()
        state["previous_task"] = self.task
        state["task_reports"] = self.task_events[-8:]
        tasks, criteria = task_options(self.memory)
        choice = self.request("planner", state, TASK_INSTRUCTIONS, criteria)
        if choice is None or self.finish(choice):
            return
        self.task_name, self.task = choice, tasks[choice]
        self.target, self.target_name = None, None
        self.task_events.append({"time": self.sim.time, "task": self.task, "status": "assigned"})
        print(
            json.dumps(
                {
                    "case": self.args.case,
                    "architecture": self.args.architecture,
                    "planner": self.args.planner,
                    "time": round(self.sim.time, 2),
                    "task": self.task_name,
                }
            ),
            flush=True,
        )

    def choose_waypoint(self):
        state = self.state()
        points, candidates, criteria = waypoint_options(
            self.scene, self.sim, self.memory, self.checked, self.attempts, self.task
        )
        state.update(
            {
                "assigned_task": self.task,
                "candidate_waypoints": candidates,
                "previous_waypoint": self.target_name,
            }
        )
        role = "navigator" if self.args.architecture == "task" else "planner"
        instructions = NAV_INSTRUCTIONS if role == "navigator" else WAYPOINT_INSTRUCTIONS
        choice = self.request(role, state, instructions, criteria)
        if choice is None or self.finish(choice):
            return
        if choice == "report_blocked":
            self.report = {"status": "blocked", "task": self.task, "waypoint": self.target_name}
            self.task_events.append({"time": self.sim.time, **self.report})
            self.task, self.target, self.target_name = None, None, None
            return
        self.assign_waypoint(choice, points)

    def move(self):
        observation = depth_observation(self.sim, self.history, self.target, self.intent)
        sent = self.sim.time
        decision = self.local.decide(observation)  # This is always policy.Jev, never the planner.
        latency = decision["latency_seconds"]
        self.sim.advance(min(latency, max(0.0, self.args.seconds - self.sim.time)))
        accepted = self.active() and latency <= 0.8 and not decision.get("error")
        if self.sim.status == "running":
            self.sim.command(decision["action"] if accepted else "brake")
        if accepted:
            self.intent = decision["intent"]
        self.calls.append(
            {
                "role": "control",
                "sent_at": sent,
                "received_at": self.sim.time,
                "accepted": accepted,
                "waypoint": self.target_name,
                "state": observation,
                **decision,
            }
        )
        self.history.append(
            {
                "position": observation["position_metres_xyz"],
                "goal_direction": observation["goal"]["direction"],
                "goal_distance": observation["goal"]["distance_metres"],
                "action": decision["action"] if accepted else "brake",
            }
        )
        self.sim.advance(min(max(0.0, 0.2 - latency), max(0.0, self.args.seconds - self.sim.time)))
        distance = norm(subtract(self.target, self.sim.position))
        if distance < self.best - 0.15:
            self.best, self.last_progress = distance, self.sim.time

    def run(self):
        try:
            while self.active() and len(self.calls) < 1100:
                if self.cost() >= self.args.budget:
                    self.sim.status = "budget_limit"
                    break
                discovery = self.observe()
                if not self.active():
                    break
                arrived = (
                    self.target is not None
                    and norm(subtract(self.target, self.sim.position)) <= 0.65
                    and norm(self.sim.velocity) <= 0.15
                )
                stalled = self.target is not None and self.sim.time - self.last_progress > 8.0
                self.report = {
                    "status": "target_discovered"
                    if discovery
                    else "arrived"
                    if arrived
                    else "stalled"
                    if stalled
                    else "traveling",
                    "waypoint": self.target_name,
                    "recognized_target": self.recognized,
                }
                if self.args.architecture == "task":
                    complete = False
                    if self.task:
                        if self.task["kind"] == "search_room":
                            progress = room_coverage(self.scene, self.checked)[self.task["room"]]
                            complete = progress["checked"] == progress["total"]
                        else:
                            complete = arrived
                    if complete:
                        self.report["status"] = (
                            "views_checked"
                            if self.task["kind"] == "search_room"
                            else "approach_complete"
                        )
                        self.task_events.append(
                            {"time": self.sim.time, "task": self.task, **self.report}
                        )
                    if self.task is None or discovery or complete:
                        self.choose_task()
                    if (
                        self.active()
                        and self.task is not None
                        and (self.target is None or arrived or stalled)
                    ):
                        self.choose_waypoint()
                elif self.target is None or arrived or stalled or discovery:
                    self.choose_waypoint()
                if self.active() and self.target is not None:
                    self.move()
            if self.sim.status == "running":
                self.sim.status = (
                    "timeout" if self.sim.time >= self.args.seconds - 1e-9 else "call_limit"
                )
        except TimeoutError:
            if self.sim.status == "running":
                self.sim.status = "timeout"
        except Exception as exc:
            self.error = type(exc).__name__
            self.sim.status = "runtime_error"
        self.sim.record()
        result = self.sim.summary()
        result.pop("final_distance_metres")
        result.update(
            {
                "architecture": self.args.architecture,
                "planner": self.args.planner,
                "case": self.args.case,
                "seed": self.args.seed,
                "mission": self.scene.mission,
                "ground_truth": ground_truth(self.scene.home),
                "portals": self.scene.portals,
                "candidate_points": self.scene.points,
                "calls": self.calls,
                "role_counts": dict(Counter(c["role"] for c in self.calls)),
                "task_events": self.task_events,
                "observed_objects": self.memory,
                "checked_views": sorted(self.checked),
                "coverage": room_coverage(self.scene, self.checked),
                "cost_usd": self.cost(),
                "error": self.error,
                "movement_model_request": MODEL,
                "movement_models_returned": sorted(
                    {
                        c.get("model", "")
                        for c in self.calls
                        if c["role"] == "control" and c.get("model")
                    }
                ),
                "target_first_seen_seconds": self.memory.get(self.scene.home.target_id, {}).get(
                    "first_seen_at"
                ),
            }
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["waypoint", "task"], required=True)
    parser.add_argument("--planner", choices=["jev", "deepseek"], required=True)
    parser.add_argument("--case", choices=["near", "far", "occluded", "absent"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=210.0)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
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
    local = Jev(load_key(args.key_file), dense_depth=True, simple_controls=True)
    try:
        result = Experiment(args, local).run()
    finally:
        local.close()
    (args.out / "episode.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "architecture",
                    "planner",
                    "case",
                    "seed",
                    "status",
                    "simulation_seconds",
                    "role_counts",
                    "cost_usd",
                    "error",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
