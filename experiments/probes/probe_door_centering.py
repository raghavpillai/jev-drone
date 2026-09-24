"""Paired Jev probe for conflicting doorway and far-waypoint references."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from experiments.representations.door_centering import (
    INSTRUCTIONS,
    REFERENCE_INSTRUCTIONS,
    describe,
)
from jev_drone.gateway import JevGateway, load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases, negatives = [], {}
    sources = [
        ("px4-house-v36-validation", "house_hidden-5311"),
        ("px4-house-v36-validation", "house_reverse-5312"),
        ("px4-house-v35-return", "house_hidden-5111"),
    ]
    for cohort, trial in sources:
        path = Path("results") / cohort / trial / "calls.jsonl"
        selected = set()
        for index, line in enumerate(path.read_text().splitlines()):
            call = json.loads(line)
            state = call["state"]
            if "control_effects" not in state:
                continue
            updated, _ = describe(
                state, call["questions"]["selection"]["criteria"], local_reference=True
            )
            phase = state.get("doorway_phase", {}).get("phase")
            task = state["task"]["name"]
            useful = [
                name
                for name, effect in updated["control_effects"].items()
                if effect.get("door_centerline_effect") == "toward_goal"
                and not effect["movement_violations"]
                and not effect["resulting_controls"]["yaw_rate_rps"]
                and not effect["resulting_controls"]["forward_mps"]
                and effect.get("far_waypoint_right_effect") != "toward_goal"
            ]
            label = f"{trial}-{task}-{index}"
            if useful and task not in selected and len(selected) < 2:
                cases.append(
                    {
                        "label": label,
                        "source": str(path),
                        "index": index,
                        "positive": True,
                        "call": call,
                    }
                )
                selected.add(task)
            if phase in ("FACE_OPENING", "CROSS_OPENING") and phase not in negatives:
                negatives[phase] = {
                    "label": label,
                    "source": str(path),
                    "index": index,
                    "positive": False,
                    "call": call,
                }
    cases.extend(negatives.values())
    protocol = {
        "cases": [{k: v for k, v in c.items() if k != "call"} for c in cases],
        "variants": ["original", "facts", "local_reference"],
        "repeats": 2,
        "scope": "Issue-focused recorded states, not a flight or reliability estimate. "
        "Includes v36 validation observations as development data for a NEW policy; "
        "v36 flights remain frozen. Inputs and physical constraints are identical; "
        "local_reference relabels lateral progress relative to the opening during centering.",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.out / "cases.json").write_text(json.dumps(cases) + "\n")
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for case in cases:
            for repeat in range(protocol["repeats"]):
                variants = protocol["variants"] if repeat == 0 else protocol["variants"][::-1]
                for variant in variants:
                    state = deepcopy(case["call"]["state"])
                    question = deepcopy(case["call"]["questions"]["selection"])
                    scored, _ = describe(state, question["criteria"], local_reference=True)
                    if variant != "original":
                        state, question["criteria"] = describe(
                            state,
                            question["criteria"],
                            local_reference=variant == "local_reference",
                        )
                        if "doorway_centering" in state:
                            question["instructions"] += INSTRUCTIONS
                            if variant == "local_reference":
                                question["instructions"] += REFERENCE_INSTRUCTIONS
                    result = gateway.choose(state, question["instructions"], question["criteria"])
                    choice = result.get("choice")
                    effect = scored["control_effects"].get(choice, {})
                    row = {
                        "case": case["label"],
                        "positive": case["positive"],
                        "variant": variant,
                        "repeat": repeat,
                        "choice": choice,
                        "error": result.get("error"),
                        "centerline_progress": effect.get("door_centerline_effect")
                        == "toward_goal",
                        "violations": effect.get("movement_violations"),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
