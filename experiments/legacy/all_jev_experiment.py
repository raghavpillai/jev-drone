"""Room search with Jev planning, recognition, navigation, recovery, looking and control."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.legacy.architecture_experiment import Experiment
from experiments.legacy.jev_navigation import LOOK, Navigation
from experiments.legacy.search_world import update_memory, visible_objects
from experiments.legacy.sim import norm, subtract
from experiments.legacy.task_planning import planning_state
from jev_drone.evidence import snapshot_sources
from jev_drone.gateway import JevGateway, load_credential


class AllJevExperiment(Experiment):
    def __init__(self, args, gateway):
        super().__init__(args, gateway)
        self.nav = Navigation(gateway, args.memory, args.limited, args.degraded, args.seed)
        self.view_angles = defaultdict(set)
        self.sightings = defaultdict(int)
        self.last_sensor_tick, self.sensor_objects = None, []
        self.recognition_signature = None

    def request(self, role, state, instructions, criteria):
        if not self.active() or self.cost() >= self.args.budget:
            raise TimeoutError("Episode limit")
        self.sim.command("brake")
        sent = self.sim.time
        result = self.local.choose(state, instructions, criteria)
        self.calls.append({"role": role, "sent_at": sent, **result})
        self.sim.advance(
            min(result["latency_seconds"], max(0.0, self.args.seconds - self.sim.time))
        )
        if result.get("error"):
            self.sim.status, self.error = "api_error", result["error"]
            return None
        return result["choice"] if self.active() else None

    def visible(self):
        tick = (int(self.sim.time * 5), self.nav.look)
        if tick == self.last_sensor_tick:
            return self.sensor_objects
        self.last_sensor_tick = tick
        visible = visible_objects(
            self.scene.home,
            self.sim.position,
            view_direction=LOOK[self.nav.look] if self.args.limited else None,
        )
        sensed = []
        rng = random.Random(self.args.seed * 100003 + tick[0] * 7 + list(LOOK).index(tick[1]))
        for obj in visible:
            if self.args.uncertain and rng.random() < 0.15:
                continue
            self.sightings[obj["id"]] += 1
            obj = dict(obj)
            if (
                self.args.uncertain
                and self.sightings[obj["id"]] <= 2
                and "wooden" in obj["appearance"]
            ):
                obj["appearance"] = (
                    "A tall wooden storage unit; shelving and contents are unclear from this view."
                )
                obj["semantic_certainty"] = "ambiguous; inspect again"
            else:
                obj["semantic_certainty"] = "clear simulated description"
            sensed.append(obj)
        self.sensor_objects = sensed
        return sensed

    def observe(self):
        visible = self.visible()
        update_memory(self.memory, visible, self.sim.position, self.sim.time)
        for name, point in self.scene.points.items():
            if norm(subtract(point["xyz"], self.sim.position)) <= 0.65:
                self.view_angles[name].add(self.nav.look)
                if (
                    not self.args.limited
                    or set(("forward", "back", "left", "right")) <= self.view_angles[name]
                ):
                    self.checked.add(name)
        signature = tuple(sorted((k, v["appearance"]) for k, v in self.memory.items()))
        changed = False
        if signature != self.recognition_signature and signature:
            self.recognition_signature = signature
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
                "Which observed object matches the requested object and room? Choose none when evidence is ambiguous; a wooden storage unit without visible shelves/books does not establish a bookcase. Reinspection can clarify.",
                {
                    "none": "No supported match",
                    **{k: v["appearance"] for k, v in self.memory.items()},
                },
            )
            if match is not None:
                changed = match != self.recognized and match != "none"
                self.recognized = match
        return changed

    def state(self):
        state = planning_state(
            self.scene,
            self.sim,
            self.memory,
            self.visible(),
            self.checked,
            self.recognized,
            self.report,
        )
        state["observation_contract"] = {
            "furniture_initially_unknown": True,
            "limited_view": self.args.limited,
            "current_look": self.nav.look,
            "object_memory_has_last_seen_times": True,
            "coverage": "Search viewpoint requires four horizontal camera looks"
            if self.args.limited
            else "Visited configured viewpoints",
        }
        for item in state["observed_objects"]:
            item["last_seen_at"] = self.memory[item["id"]]["last_seen_at"]
            item["visible_now"] = item["id"] in {o["id"] for o in state["visible_now"]}
        if self.args.memory != "history":
            state["observed_space"] = self.nav.map.describe(self.sim.position, self.sim.time)
            state["recovery_attempts"] = self.nav.events[-8:]
        return state

    def choose_waypoint(self):
        # Continue a Jev-assigned inspection until Jev has looked around it.
        if (
            self.args.limited
            and self.target_name
            and self.target_name.startswith("search_")
            and self.target_name not in self.checked
            and norm(subtract(self.target, self.sim.position)) <= 0.65
        ):
            return
        super().choose_waypoint()

    def move(self):
        if self.nav.target != tuple(self.target):
            self.nav.target = tuple(self.target)
            self.nav.recovery_target = None
            self.nav.best = norm(subtract(self.target, self.sim.position))
            self.nav.last_progress = self.sim.time
            self.nav.intent = "direct"
        recovery = self.nav.recovery(self.sim, self.target, self.report)
        if recovery:
            self.sim.command("brake")
            self.calls.append({"role": "recovery", "sent_at": self.sim.time, **recovery})
            self.sim.advance(
                min(recovery["latency_seconds"], max(0.0, self.args.seconds - self.sim.time))
            )
        if not self.active():
            return
        missing = []
        if (
            self.target_name
            and self.target_name.startswith("search_")
            and norm(subtract(self.target, self.sim.position)) <= 0.65
        ):
            missing = sorted(
                set(("forward", "back", "left", "right")) - self.view_angles[self.target_name]
            )
        sent = self.sim.time
        decision = self.nav.decide(
            self.sim, self.target, {"task": self.task, "missing_views_here": missing}
        )
        latency = decision["latency_seconds"]
        self.sim.advance(min(latency, max(0.0, self.args.seconds - self.sim.time)))
        accepted = self.active() and latency <= 0.8 and not decision.get("error")
        if self.sim.status == "running":
            self.sim.command(decision["action"] if accepted else "brake")
        self.nav.accept(self.sim, decision, accepted, self.target)
        self.calls.append(
            {
                "role": "control",
                "sent_at": sent,
                "accepted": accepted,
                "waypoint": self.target_name,
                **decision,
            }
        )
        self.sim.advance(min(max(0.0, 0.2 - latency), max(0.0, self.args.seconds - self.sim.time)))
        distance = norm(subtract(self.target, self.sim.position))
        if distance < self.best - 0.15:
            self.best, self.last_progress = distance, self.sim.time

    def run(self):
        result = super().run()
        result.update(
            memory_mode=self.args.memory,
            limited_view=self.args.limited,
            uncertain_semantics=self.args.uncertain,
            degraded_depth=self.args.degraded,
            provider=self.local.provider,
            recovery_events=self.nav.events,
            observed_map_cells=len(self.nav.map.cells),
            camera_views={k: sorted(v) for k, v in self.view_angles.items()},
            movement_model_request=self.local.model,
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openrouter", "vercel"], default="openrouter")
    parser.add_argument("--memory", choices=["history", "map", "recovery"], default="recovery")
    parser.add_argument("--case", choices=["near", "far", "occluded", "absent"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=210.0)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--limited", action="store_true")
    parser.add_argument("--uncertain", action="store_true")
    parser.add_argument("--degraded", action="store_true")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.architecture, args.planner = "task", "jev"
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
    gateway = JevGateway(args.provider, key)
    try:
        result = AllJevExperiment(args, gateway).run()
    finally:
        gateway.close()
    (args.out / "episode.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "case",
                    "seed",
                    "status",
                    "simulation_seconds",
                    "role_counts",
                    "cost_usd",
                    "error",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
