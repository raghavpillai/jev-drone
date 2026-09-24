"""Post-failure development probe: scope altitude replanning to near-goal states."""

import json
from copy import deepcopy
from pathlib import Path

from experiments.representations.altitude_access import INSTRUCTIONS, describe
from jev_drone.gateway import JevGateway, load_credential

root = Path("results/px4-house-v40-scope-probe")
root.mkdir(parents=True, exist_ok=False)
cases = json.loads(Path("results/px4-house-v39-altitude-probe/cases.json").read_text())
path = Path("results/px4-house-v39-validation/house_reverse-5612/calls.jsonl")
seen = set()
for index, line in enumerate(path.read_text().splitlines()):
    try:
        c = json.loads(line)
    except ValueError:
        continue
    s = c["state"]
    fact = s.get("altitude_access", {})
    if (
        c.get("answers", {}).get("selection", {}).get("choice") != "brake_and_replan"
        or fact.get("horizontal_reached") is not False
    ):
        continue
    task = s["task"]["name"]
    if task in seen:
        continue
    seen.add(task)
    c = deepcopy(c)
    c["state"].pop("altitude_access", None)
    q = c["questions"]["selection"]
    q["instructions"] = q["instructions"].replace(INSTRUCTIONS, "")
    suffix = " The requested height correction is blocked by observed clearance; stopping and asking for another approach is appropriate."
    assert q["criteria"]["brake_and_replan"].endswith(suffix)
    q["criteria"]["brake_and_replan"] = q["criteria"]["brake_and_replan"].removesuffix(suffix)
    cases.append(
        dict(
            label="far-loop-" + task, kind="far_replan_loop", source=str(path), index=index, call=c
        )
    )
    if len(seen) == 4:
        break
(root / "cases.json").write_text(json.dumps(cases) + "\n")
(root / "protocol.json").write_text(
    json.dumps(
        {
            "cases": [{k: v for k, v in c.items() if k != "call"} for c in cases],
            "variants": ["baseline", "broad_altitude", "near_goal_only"],
            "repeats": 2,
            "scope": "Post-failure development on historical v34-v38 cases plus four current v39 replanning states. V39 flights stay unchanged. This does not create a new native policy or validate mission success; v39 data are now development sources for the proposed scope correction.",
        },
        indent=2,
    )
    + "\n"
)
gateway = JevGateway(
    "openrouter", load_credential("openrouter", Path(".env")), journal=root / "calls.jsonl"
)
rows = []
try:
    for case in cases:
        for repeat in range(2):
            for variant in (
                ["baseline", "broad_altitude", "near_goal_only"]
                if repeat == 0
                else ["near_goal_only", "broad_altitude", "baseline"]
            ):
                s = deepcopy(case["call"]["state"])
                q = deepcopy(case["call"]["questions"]["selection"])
                if (
                    variant == "broad_altitude"
                    or variant == "near_goal_only"
                    and s["horizontal_distance"] <= s["goal_tolerance_m"]["horizontal"]
                ):
                    s, q["criteria"] = describe(s, q["criteria"])
                    anchor = "\nRead mission_progress before refining the waypoint."
                    assert anchor in q["instructions"]
                    q["instructions"] = q["instructions"].replace(anchor, INSTRUCTIONS + anchor, 1)
                r = gateway.choose(s, q["instructions"], q["criteria"])
                choice = r.get("choice")
                e = s["control_effects"].get(choice, {})
                row = dict(
                    case=case["label"],
                    kind=case["kind"],
                    variant=variant,
                    repeat=repeat,
                    choice=choice,
                    violations=e.get("movement_violations"),
                    travel=e.get("travel_effect"),
                    turn=e.get("turn_effect"),
                    error=r.get("error"),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
finally:
    gateway.close()
    (root / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
