"""Paired live Jev decisions on recorded braking failures; no vehicle commands."""

import argparse
import json
from pathlib import Path

from jev_drone.control.decision_instructions import BRAKING, braking_facts
from jev_drone.gateway import JevGateway, load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Sample each movement axis as well as braking failures",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for path in sorted(Path("results/px4-joystick-final-validation").glob("*/result.json")):
        result = json.loads(path.read_text())
        violations = {v["time"] for v in result["violations"]}
        controls = [c for c in result["calls"] if c["role"] == "control"]
        cases.extend(
            (path.parent.name, "braking_failure", c) for c in controls if c["time"] in violations
        )
        # Include successful movement to check that the instructions do not just stop everything.
        nominal = [
            c
            for c in controls
            if c.get("accepted")
            and c["choice"] in ("forward_full", "strafe_left_slow", "pan_left", "up")
            and not c["state"]["control_effects"][c["choice"]]["movement_violations"]
        ]
        if args.diverse:
            nominal = []
            for name in (
                "forward_full",
                "strafe_left_slow",
                "strafe_right_slow",
                "pan_left",
                "pan_right",
                "up",
                "down",
            ):
                call = next(
                    (
                        c
                        for c in controls
                        if c.get("accepted")
                        and c["choice"] == name
                        and not c["state"]["control_effects"][name]["movement_violations"]
                    ),
                    None,
                )
                if call:
                    nominal.append(call)
        cases.extend(
            (path.parent.name, "nominal", c) for c in (nominal if args.diverse else nominal[:2])
        )
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for trial, kind, call in cases:
            for repeat in range(2):
                for variant in ("baseline", "braking_facts"):
                    state = call["state"]
                    question = call["questions"]["selection"]
                    instructions, criteria = question["instructions"], question["criteria"]
                    if variant == "braking_facts":
                        state, criteria = braking_facts(state, criteria)
                        instructions += BRAKING
                    result = gateway.choose(state, instructions, criteria)
                    choice = result.get("choice")
                    effect = state["control_effects"].get(choice, {})
                    row = dict(
                        trial=trial,
                        time=call["time"],
                        kind=kind,
                        repeat=repeat,
                        variant=variant,
                        choice=choice,
                        violations=effect.get("movement_violations"),
                        nonzero=any(effect.get("resulting_controls", {}).values()),
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
