"""Compare Jev's recorded unsafe axis changes with explicit transition choices."""

import argparse
import copy
import json
from pathlib import Path

from jev_drone.control.policy import CHANNELS, movement_reasons
from jev_drone.gateway import JevGateway, load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    calls = [
        c
        for c in result["calls"]
        if c["time"] in {v["time"] for v in result["violations"]} and c["role"] == "control"
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for index, call in enumerate(calls):
            for variant in ("baseline", "explicit_transition"):
                state = copy.deepcopy(call["state"])
                q = copy.deepcopy(call["questions"]["selection"])
                if variant == "explicit_transition":
                    for direction, speed in [("up", 0.25), ("down", -0.25)]:
                        values = {**dict.fromkeys(CHANNELS, 0.0), "up_mps": speed}
                        patch = {
                            key: value
                            for key, value in values.items()
                            if state["current_controls"]["body_controls"][key] != value
                        }
                        reasons = movement_reasons(values, state["sensors"])
                        name = "straight_" + direction
                        state["control_effects"][name] = {
                            "patch": patch,
                            "resulting_controls": values,
                            "movement_violations": reasons,
                            "turn_effect": "hold_aligned_heading",
                            "travel_effect": {
                                "forward": "stopped",
                                "right": "stopped",
                                "up": "toward_goal"
                                if speed * state["body_goal_offset_m"]["up"] > 0
                                else "away_from_goal",
                            },
                        }
                        q["criteria"][name] = (
                            f"Apply {patch}: stop horizontal motion and pan, move only {direction}. Movement violations: {reasons}."
                        )
                    q["instructions"] += (
                        "\nFor a transition between axes, use straight_up or straight_down to release horizontal controls and start vertical motion together. Plain up/down preserves forward and strafe; do not accidentally keep an axis that already reached its goal.\n"
                    )
                response = gateway.choose(state, q["instructions"], q["criteria"])
                choice = response.get("choice")
                row = dict(
                    index=index,
                    variant=variant,
                    choice=choice,
                    violations=state["control_effects"][choice]["movement_violations"]
                    if choice
                    else None,
                    latency_seconds=response["latency_seconds"],
                    error=response.get("error"),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        gateway.close()
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
