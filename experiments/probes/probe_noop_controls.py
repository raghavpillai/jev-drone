"""Recorded-state comparison for repeated releases of already stopped controls."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential

NOOP = """
Read neutral_and_settled_away_from_goal. When true, the vehicle is ALREADY stopped,
the position goal remains unfinished, and a permitted translation toward it exists.
Repeated stop_pan/stop_forward/other releases cannot make position progress when
all resulting controls remain zero. Choose a permitted movement that makes progress
instead of repeatedly releasing an already zero axis. This does not apply during
measured braking, arrival, or a ready inspection/docking report. Safety still wins.
"""
PATH = """
Unknown goal_path_clearance_m means the whole direct goal segment is not certified.
It does not prohibit a short body-axis movement whose own movement_violations is
empty and travel_effect points toward the goal. Use the observed command corridor,
then reassess depth. If already facing a required doorway heading within 8 degrees,
do not repeatedly stop_pan or wait for an exact zero-degree error before beginning
a permitted centered crossing. You still choose every movement.
"""


def describe(state, criteria):
    state, criteria = deepcopy(state), dict(criteria)
    observation, effects = state["sensors"], state["control_effects"]
    progress = state.get("mission_progress", {})
    eligible = (
        not any(state["current_controls"]["body_controls"].values())
        and observation["speed"] <= 0.1
        and abs(observation["yaw_rate_rps"]) <= 0.1
        and not state["at_position"]
        and not progress.get("target_in_inspection_range")
        and not progress.get("dock_in_range")
        and any(
            not e["movement_violations"] and "toward_goal" in e["travel_effect"].values()
            for e in effects.values()
            if "travel_effect" in e
        )
    )
    state["neutral_and_settled_away_from_goal"] = eligible
    if eligible:
        for name, effect in effects.items():
            if (
                not any(effect["resulting_controls"].values())
                and not name.startswith("detour_")
                and name != "brake_and_replan"
            ):
                effect["position_progress"] = "already_stopped_no_position_progress"
                criteria[name] += (
                    " ALREADY STOPPED: this makes no position progress. A permitted translation toward the unfinished goal exists; do not repeat an already satisfied release."
                )
    return state, criteria


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = []
    for trial, kind, count in (
        ("sequence-4933", "stationary_loop", 3),
        ("controller-4910", "braking", 2),
        ("controller-4910", "at_position", 1),
    ):
        root = Path("results/px4-recovery-followup-v24") / trial
        result = json.loads((root / "result.json").read_text())
        journal = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
        found, last = [], -100.0
        for index, (call, raw) in enumerate(zip(result["calls"], journal)):
            state = raw["state"]
            if call["role"] != "control" or raw.get("error") or call["time"] - last < 5.0:
                continue
            if kind == "stationary_loop":
                match = (
                    call["time"] > 300
                    and call.get("choice") == "stop_pan"
                    and describe(state, raw["questions"]["selection"]["criteria"])[0][
                        "neutral_and_settled_away_from_goal"
                    ]
                )
            elif kind == "braking":
                match = state.get("neutral_controls_still_braking", False)
            else:
                match = state["at_position"]
            if match:
                found.append((trial, kind, index, call["time"], raw))
                last = call["time"]
            if len(found) == count:
                break
        if len(found) != count:
            raise ValueError(f"Missing probe states for {trial}/{kind}")
        selected.extend(found)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_text(
        json.dumps(
            {
                "states": [
                    {"trial": t, "kind": k, "call_index": i, "sim_time": s}
                    for t, k, i, s, _ in selected
                ],
                "variants": ["baseline", "noop_facts", "noop_and_path"],
                "repeats": 2,
                "noop_instruction": NOOP,
                "path_instruction": PATH,
                "limitation": "Recorded-state choice behavior, not flight success. All movement choices remain available.",
            },
            indent=2,
        )
        + "\n"
    )
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for trial, kind, index, time, raw in selected:
            for repeat in range(2):
                variants = ["baseline", "noop_facts", "noop_and_path"]
                if repeat:
                    variants.reverse()
                for variant in variants:
                    question = raw["questions"]["selection"]
                    state, criteria, instructions = (
                        raw["state"],
                        question["criteria"],
                        question["instructions"],
                    )
                    if variant != "baseline":
                        state, criteria = describe(state, criteria)
                        instructions += NOOP + (PATH if variant == "noop_and_path" else "")
                    response = gateway.choose(state, instructions, criteria)
                    choice = response.get("choice")
                    effect = state["control_effects"].get(choice, {})
                    row = dict(
                        trial=trial,
                        kind=kind,
                        call_index=index,
                        sim_time=time,
                        repeat=repeat,
                        variant=variant,
                        choice=choice,
                        error=response.get("error"),
                        violations=effect.get("movement_violations"),
                        nonzero=any(effect.get("resulting_controls", {}).values()),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
