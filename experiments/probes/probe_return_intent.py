"""Paired Jev return decisions on recorded failed missions, without flying."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.planner_modes import eligible
from jev_drone.planning.return_intent import INSTRUCTIONS, describe

SOURCES = [
    ("px4-house-v30-survey", "house_hidden-5111", [710, 816, 974, 1041, 1120]),
    ("px4-house-v29-transitions", "house_hidden-5111", [1099]),
    ("px4-house-v31-progress", "house_search-5111", [203, 261]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--scope",
        action="store_true",
        help="Exclude object search/approach categories from dock objectives.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for cohort, trial, times in SOURCES:
        root = Path("results") / cohort / trial
        result = json.loads((root / "result.json").read_text())
        journal = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
        modes = [
            i
            for i, c in enumerate(result["calls"])
            if c["role"] == "planner_mode" and c["state"].get("stage") == 1
        ]
        for time in times:
            index = min(modes, key=lambda i: abs(result["calls"][i]["time"] - time))
            assert result["calls"][index + 1]["role"] == "planner"
            cases.append(
                (
                    cohort + "-" + str(round(result["calls"][index]["time"])),
                    journal[index],
                    journal[index + 1],
                )
            )
    protocol = {
        "sources": SOURCES,
        "cases": [label for label, _, _ in cases],
        "repeats": 2,
        "variants": ["original", "room_return"],
        "exclude_marker_tasks_during_dock": args.scope,
        "scope": "Same recorded concrete options and sensing; change return intent facts/instructions and optionally exclude inapplicable marker task categories during dock. Each arm calls Jev for task kind and then concrete task. These are decision probes, not native mission passes.",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for label, mode_call, task_call in cases:
            for repeat in range(2):
                for variant in protocol["variants"] if repeat == 0 else protocol["variants"][::-1]:
                    state = deepcopy(mode_call["state"])
                    modeq = deepcopy(mode_call["questions"]["selection"])
                    taskq = deepcopy(task_call["questions"]["selection"])
                    # Recover all concrete candidates; the original call only offered its selected family.
                    criteria = {
                        name: taskq["criteria"].get(name, json.dumps(option))
                        for name, option in state["options"].items()
                    }
                    if variant == "room_return":
                        state, criteria = describe(state, criteria)
                        modeq["instructions"] = (
                            modeq["instructions"].split("\nFor return_to_launch,")[0] + INSTRUCTIONS
                        )
                        modeq["criteria"]["retrace"] = (
                            "Returning and local approach is blocked: consider a nearby observed outbound position as a local bypass; do not replay the entire search history."
                        )
                        if args.scope:
                            modeq["criteria"] = {
                                name: text
                                for name, text in modeq["criteria"].items()
                                if name not in ("search", "approach")
                            }
                        old = "On return_to_launch, retrace options are positions YOU already flew through on the"
                        end = "If launch is nearby with a fully observed clear direct segment, return_launch is\nappropriate. Do not return to a higher index after completing a lower one.\n"
                        if old in taskq["instructions"]:
                            start = taskq["instructions"].index(old)
                            finish = taskq["instructions"].index(end, start) + len(end)
                            taskq["instructions"] = (
                                taskq["instructions"][:start] + taskq["instructions"][finish:]
                            )
                        taskq["instructions"] += INSTRUCTIONS
                    mode = gateway.choose(state, modeq["instructions"], modeq["criteria"])
                    kind = mode.get("choice")
                    task = None
                    choice = None
                    if kind:
                        state["selected_task_kind"] = kind
                        task_instructions = (
                            taskq["instructions"].split("\nYou selected task kind ")[0]
                            + "\nYou selected task kind "
                            + kind
                            + ". Select the concrete task from current evidence."
                        )
                        task = gateway.choose(
                            state,
                            task_instructions,
                            {n: c for n, c in criteria.items() if eligible(n, kind)},
                        )
                        choice = task.get("choice")
                    option = state["options"].get(choice, {})
                    row = {
                        "case": label,
                        "repeat": repeat,
                        "variant": variant,
                        "current_room": state["sensors"]["room"],
                        "launch_room": state["ACTIVE_OBJECTIVE"]["room"],
                        "kind": kind,
                        "choice": choice,
                        "destination_room": option.get("leads_to_room", option.get("room")),
                        "observed_endpoint_unsafe": option.get("observed_endpoint_unsafe"),
                        "error": mode.get("error") or (task or {}).get("error"),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
