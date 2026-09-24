"""Paired live Jev decisions at an observed accidental room reversal."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.room_transition import INSTRUCTIONS, describe
from jev_drone.world.layout import HOUSE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    path = Path("results/px4-house-v28-door-phase/house_hidden-5111/calls.jsonl")
    calls = [json.loads(x) for x in path.read_text().splitlines()]
    cases = []
    for call in calls:
        state = call["state"]
        if "selected_task_kind" not in state:
            continue
        previous = next(
            (
                t
                for t in reversed(state.get("recent_tasks", []))
                if t["name"].startswith("cross_") and t["outcome"] == "arrived"
            ),
            None,
        )
        if previous and previous.get("leads_to_room") == state["sensors"]["room"]:
            cases.append(call)
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for index, call in enumerate(cases[:4]):
            for repeat in range(3):
                for variant in (
                    ("baseline", "completed_transition")
                    if repeat % 2 == 0
                    else ("completed_transition", "baseline")
                ):
                    state = deepcopy(call["state"])
                    q = deepcopy(call["questions"]["selection"])
                    annotated, _ = describe(state, q["criteria"], HOUSE)
                    reverse = {
                        name
                        for name, option in annotated["options"].items()
                        if option.get("reverses_completed_crossing")
                    }
                    if variant == "completed_transition":
                        state, q["criteria"] = describe(state, q["criteria"], HOUSE)
                        q["instructions"] += INSTRUCTIONS
                    r = gateway.choose(state, q["instructions"], q["criteria"])
                    row = dict(
                        index=index,
                        repeat=repeat,
                        variant=variant,
                        room=state["sensors"]["room"],
                        choice=r.get("choice"),
                        backtracks=r.get("choice") in reverse,
                        error=r.get("error"),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
