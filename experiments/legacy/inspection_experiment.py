"""Follow-up: supply the local Jev controller the observed object it must inspect."""

import argparse
import json
from pathlib import Path

from experiments.legacy.all_jev_experiment import AllJevExperiment
from experiments.legacy.jev_navigation import Navigation
from experiments.legacy.sim import subtract
from jev_drone.evidence import snapshot_sources
from jev_drone.gateway import JevGateway, load_credential


class InspectionNavigation(Navigation):
    observed_target = None

    def decide(self, sim, target, extra=None):
        extra = dict(extra or {})
        if self.observed_target:
            obj = self.observed_target
            extra["target_observation"] = {
                "appearance": obj["appearance"],
                "last_seen_at": obj["last_seen_at"],
                "surface_offset_forward_left_up_m": [
                    round(v, 2) for v in subtract(obj["visible_surface_xyz"], sim.position)
                ],
                "instruction": "This is the remembered object, not the flight waypoint. When near the approach waypoint, aim the camera toward this observed object to confirm it. Do not fly into the surface.",
            }
        return super().decide(sim, target, extra)


class InspectionExperiment(AllJevExperiment):
    def __init__(self, args, gateway):
        super().__init__(args, gateway)
        self.nav = InspectionNavigation(
            gateway, args.memory, args.limited, args.degraded, args.seed
        )

    def request(self, role, state, instructions, criteria):
        if role == "planner":
            instructions += "\nIf the recognized target is nearby but not currently visible, assign its approach task again so the local Jev can aim the camera toward the remembered object and inspect it. Do not abandon an identified target merely because the camera currently faces away."
        return super().request(role, state, instructions, criteria)

    def move(self):
        self.nav.observed_target = self.memory.get(self.recognized)
        super().move()

    def run(self):
        result = super().run()
        result["inspection_context"] = True
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["near", "far", "occluded", "absent"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--memory", choices=["history", "map", "recovery"], default="history")
    parser.add_argument("--provider", choices=["openrouter", "vercel"], default="openrouter")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.limited, args.uncertain, args.degraded = True, True, False
    args.architecture, args.planner = "task", "jev"
    args.seconds, args.budget = 210.0, 0.25
    key = load_credential(args.provider, args.key_file)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    gateway = JevGateway(args.provider, key)
    try:
        result = InspectionExperiment(args, gateway).run()
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
