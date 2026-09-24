"""Test whether optional travel heading avoids unnecessary Jev turning."""

import argparse
import json
from pathlib import Path

from jev_drone.control.decision_instructions import BRAKING, TRAVEL, braking_facts, heading_facts
from jev_drone.gateway import JevGateway, load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-braking", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for path in sorted(Path("results/px4-joystick-final-validation").glob("*/result.json")):
        calls = json.loads(path.read_text())["calls"]
        candidates = []
        for call in calls:
            state = call["state"]
            if (
                call["role"] != "control"
                or not call.get("accepted")
                or state["at_position"]
                or state.get("required_heading_degrees") is not None
                or state["active_detour"]
                or state["goal_path_status"] != "OBSERVED_CLEAR"
                or state["horizontal_distance"] < 0.5
            ):
                continue
            if state["control_effects"][call["choice"]]["resulting_controls"]["yaw_rate_rps"]:
                candidates.append(call)
        cases.extend((path.parent.name, c) for c in candidates[:: max(1, len(candidates) // 3)][:3])
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for trial, call in cases:
            for repeat in range(2):
                for variant in ("baseline", "optional_heading"):
                    state = call["state"]
                    question = call["questions"]["selection"]
                    instructions, criteria = question["instructions"], question["criteria"]
                    if variant == "optional_heading":
                        state, criteria = heading_facts(state, criteria)
                        instructions += TRAVEL
                    if args.with_braking:
                        state, criteria = braking_facts(state, criteria)
                        instructions += BRAKING
                    result = gateway.choose(state, instructions, criteria)
                    effect = state["control_effects"].get(result.get("choice"), {})
                    values = effect.get("resulting_controls", {})
                    row = dict(
                        trial=trial,
                        time=call["time"],
                        repeat=repeat,
                        variant=variant,
                        choice=result.get("choice"),
                        braking_instructions=args.with_braking,
                        violations=effect.get("movement_violations"),
                        pan=bool(values.get("yaw_rate_rps")),
                        translation=any(
                            values.get(k, 0) for k in ("forward_mps", "right_mps", "up_mps")
                        ),
                        travel_effect=effect.get("travel_effect"),
                        latency_seconds=result["latency_seconds"],
                        error=result.get("error"),
                        cost_usd=result.get("usage", {}).get("cost", 0),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
