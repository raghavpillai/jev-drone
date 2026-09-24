"""Paired Jev choices on actual doorway states; never issues flight controls."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from jev_drone.control.doorway_phase import INSTRUCTIONS, describe
from jev_drone.gateway import JevGateway, load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path("results/px4-house-v26-development")
    cases = []
    for name in ("house_far-5101", "house_reverse-5102"):
        calls = [json.loads(x) for x in (root / name / "calls.jsonl").read_text().splitlines()]
        for kind in ("misaligned", "aligned"):
            eligible = []
            for c in calls:
                s = c["state"]
                a = s.get("doorway_alignment")
                if a is None or s.get("active_detour") or not c.get("answers"):
                    continue
                bad = abs(a["altitude_error_m"]) > 0.25 and abs(a["heading_error_degrees"]) <= 8
                good = (
                    abs(a["altitude_error_m"]) < 0.1
                    and abs(a["centerline_offset_m"]) < 0.15
                    and abs(a["heading_error_degrees"]) < 5
                )
                if (kind == "misaligned" and bad) or (kind == "aligned" and good):
                    eligible.append(c)
            cases.extend((name, kind, c) for c in eligible[:: max(1, len(eligible) // 3)][:3])
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for index, (name, kind, call) in enumerate(cases):
            for repeat in range(2):
                for variant in (
                    ("baseline", "door_phase") if repeat == 0 else ("door_phase", "baseline")
                ):
                    state = deepcopy(call["state"])
                    q = deepcopy(call["questions"]["selection"])
                    if variant == "door_phase":
                        state, q["criteria"] = describe(state, q["criteria"])
                        q["instructions"] += INSTRUCTIONS
                    r = gateway.choose(state, q["instructions"], q["criteria"])
                    choice = r.get("choice")
                    effect = state["control_effects"].get(choice, {})
                    row = dict(
                        index=index,
                        trial=name,
                        kind=kind,
                        repeat=repeat,
                        variant=variant,
                        choice=choice,
                        phase=state.get("doorway_phase"),
                        violations=effect.get("movement_violations"),
                        vertical_toward=effect.get("travel_effect", {}).get("up") == "toward_goal",
                        error=r.get("error"),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
