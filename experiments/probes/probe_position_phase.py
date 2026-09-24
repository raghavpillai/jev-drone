"""Replay vertical-waypoint turning-loop requests with explicit position phase."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from jev_drone.control.goal_phase import INSTRUCTIONS, describe
from jev_drone.gateway import JevGateway, load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    source = Path("results/px4-house-v32-edges/house_search_absent-5111/calls.jsonl")
    eligible = []
    with source.open() as stream:
        for line in stream:
            if '"control_effects"' not in line:
                continue
            call = json.loads(line)
            s = call["state"]
            if (
                s.get("task", {}).get("name") == "view_workshop_edge_west_2.2"
                and not s.get("active_detour")
                and s["horizontal_distance"] <= s["goal_tolerance_m"]["horizontal"]
                and abs(s["body_goal_offset_m"]["up"]) > 0.4
            ):
                eligible.append(call)
    cases = eligible[:: max(1, len(eligible) // 4)][:4]
    (a.out / "protocol.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "cases": len(cases),
                "variants": ["original", "position_phase"],
                "scope": "Recorded vertical-goal states, unchanged action universe and motion constraints. Compare safe vertical progress to repeated pan. Decision probe only.",
            },
            indent=2,
        )
        + "\n"
    )
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=a.out / "calls.jsonl"
    )
    rows = []
    try:
        for index, call in enumerate(cases):
            for variant in (
                ("original", "position_phase") if index % 2 == 0 else ("position_phase", "original")
            ):
                s = deepcopy(call["state"])
                q = deepcopy(call["questions"]["selection"])
                if variant == "position_phase":
                    s, q["criteria"] = describe(s, q["criteria"])
                    q["instructions"] += INSTRUCTIONS
                response = gateway.choose(s, q["instructions"], q["criteria"])
                choice = response.get("choice")
                effect = s["control_effects"].get(choice, {})
                row = dict(
                    index=index,
                    variant=variant,
                    choice=choice,
                    error=response.get("error"),
                    vertical_progress=effect.get("travel_effect", {}).get("up") == "toward_goal",
                    violations=effect.get("movement_violations"),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (a.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
