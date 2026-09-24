"""Test whether the parent inspection heading distracts from a vertical bypass leg."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential

ALTITUDE = """
Read bypass_position_components. If horizontal_reached is true and
altitude_reached is false, horizontal travel is finished but the BYPASS POSITION
is not reached. Finish its altitude before facing the parent inspection item.
Choose allowed up/down movement toward the signed remaining altitude; up/down
also releases pan and horizontal controls. If already stopped, do not repeatedly
brake instead of making this permitted vertical progress. Never violate depth
or braking constraints. The parent item's heading is deferred until this bypass
position is reached. These facts do not select a control for you.
"""


def candidate(state, criteria):
    state, criteria = deepcopy(state), dict(criteria)
    if (
        state.get("active_detour")
        and state.get("required_heading_degrees") is None
        and state["horizontal_distance"] <= 0.35
    ):
        state["requested_heading_degrees"] = None
        state["goal_bearing_error_degrees"] = 0.0
        for name, effect in state["control_effects"].items():
            previous = effect["turn_effect"]
            effect["turn_effect"] = (
                "turn_away_from_alignment"
                if effect["resulting_controls"]["yaw_rate_rps"]
                else "hold_aligned_heading"
            )
            criteria[name] = criteria[name].replace(previous, effect["turn_effect"])
    return state, criteria


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--altitude-facts", action="store_true")
    args = parser.parse_args()
    selected = []
    for name in (
        "px4-recovery-validation-v23/occluded-4932",
        "px4-recovery-evidence-v20/occluded-4872",
    ):
        root = Path("results") / name
        result = json.loads((root / "result.json").read_text())
        journal = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
        found, last_time = [], -100.0
        for index, (call, raw) in enumerate(zip(result["calls"], journal)):
            state = raw["state"]
            if (
                call["role"] == "control"
                and not raw.get("error")
                and state.get("active_detour")
                and state["task"].get("look_position")
                and state.get("required_heading_degrees") is None
                and state["horizontal_distance"] <= 0.25
                and abs(state["body_goal_offset_m"]["up"]) > 0.5
                and call["time"] - last_time >= 2.5
            ):
                direction = "up" if state["body_goal_offset_m"]["up"] > 0 else "down"
                if state["control_effects"][direction]["movement_violations"]:
                    continue
                found.append((name, index, call["time"], direction, raw))
                last_time = call["time"]
            if len(found) == 3:
                break
        if len(found) != 3:
            raise ValueError(f"Not enough separated eligible states from {name}")
        selected.extend(found)
    args.out.mkdir(parents=True, exist_ok=False)
    variant_name = (
        "defer_heading_and_altitude_facts" if args.altitude_facts else "defer_parent_heading"
    )
    (args.out / "protocol.json").write_text(
        json.dumps(
            {
                "states": [
                    {"trial": n, "call_index": i, "sim_time": t, "vertical_direction": d}
                    for n, i, t, d, _ in selected
                ],
                "repeats": 2,
                "variants": ["baseline", variant_name],
                "additional_instruction": ALTITUDE if args.altitude_facts else None,
                "limitation": "Recorded-state control choice comparison, not flight completion evidence.",
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
        for name, index, time, direction, raw in selected:
            for repeat in range(2):
                variants = ["baseline", variant_name]
                if repeat:
                    variants.reverse()
                for variant in variants:
                    question = raw["questions"]["selection"]
                    state, criteria = raw["state"], question["criteria"]
                    instructions = question["instructions"]
                    if variant != "baseline":
                        state, criteria = candidate(state, criteria)
                        if args.altitude_facts:
                            state["bypass_position_components"] = {
                                "horizontal_reached": state["horizontal_distance"]
                                <= state["goal_tolerance_m"]["horizontal"],
                                "altitude_reached": abs(state["body_goal_offset_m"]["up"])
                                <= state["goal_tolerance_m"]["vertical"],
                                "remaining_altitude_m": state["body_goal_offset_m"]["up"],
                                "parent_inspection_heading_deferred": True,
                            }
                            instructions += ALTITUDE
                    response = gateway.choose(state, instructions, criteria)
                    choice = response.get("choice")
                    row = dict(
                        trial=name,
                        call_index=index,
                        sim_time=time,
                        repeat=repeat,
                        variant=variant,
                        choice=choice,
                        vertical_progress=choice in (direction, direction + "_creep"),
                        violations=state["control_effects"][choice]["movement_violations"]
                        if choice
                        else None,
                        error=response.get("error"),
                        latency_seconds=response["latency_seconds"],
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
