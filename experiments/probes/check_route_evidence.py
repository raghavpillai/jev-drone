"""Compare detour decisions with later-leg occupancy from earlier camera frames."""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from experiments.probes.check_door_memory import recorded_cloud
from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.detour_route import INSTRUCTIONS, observed_obstacles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    path = Path("results/px4-recovery-cycle-v18/vertical-4874")
    result = json.loads((path / "result.json").read_text())
    snapshots = sorted((path / "result-frames").glob("*.npz"))
    cases = []
    for call in result["calls"]:
        if call["role"] != "control" or not call.get("choice", "").startswith("detour_"):
            continue
        if cases and call["time"] - cases[-1][0]["time"] < 10:
            continue
        available = [p for p in snapshots if float(p.stem) <= call["time"]]
        if not available:
            continue
        snapshot = available[-1]
        frame = SimpleNamespace(clearance_cloud=recorded_cloud(snapshot, result["world"]["hfov"]))
        facts = {
            name: observed_obstacles(
                effect["planned_route"], call["state"]["sensors"]["position_estimate"], [frame]
            )
            for name, effect in call["state"]["control_effects"].items()
            if "planned_route" in effect
        }
        if any(leg["observed_blocked"] for leg in facts[call["choice"]]):
            cases.append((call, facts, snapshot))
        if len(cases) == 4:
            break
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for call, facts, snapshot in cases:
            for repeat in range(2):
                for variant in ("baseline", "route_evidence"):
                    state = deepcopy(call["state"])
                    question = call["questions"]["selection"]
                    criteria, instructions = dict(question["criteria"]), question["instructions"]
                    if variant == "route_evidence":
                        instructions += INSTRUCTIONS
                        for name, legs in facts.items():
                            state["control_effects"][name]["route_obstacles"] = legs
                            criteria[name] += (
                                " Observed route obstacles: "
                                + json.dumps(legs)
                                + ". Prefer a route without observed blocked legs; unseen space is not guaranteed clear."
                            )
                    response = gateway.choose(state, instructions, criteria)
                    choice = response.get("choice")
                    row = dict(
                        time=call["time"],
                        frame=str(snapshot),
                        repeat=repeat,
                        variant=variant,
                        choice=choice,
                        selected_observed_blocked_route=any(
                            leg["observed_blocked"] for leg in facts.get(choice, [])
                        ),
                        violations=state["control_effects"]
                        .get(choice, {})
                        .get("movement_violations"),
                        error=response.get("error"),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
