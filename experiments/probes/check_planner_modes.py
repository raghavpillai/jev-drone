"""Offline two-decision Jev planner experiment; never sends vehicle commands."""
import argparse
import json
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.planner_modes import MODES, INSTRUCTIONS, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [Path("results/px4-autonomy-transit-v12/occluded-4753/result.json"),
             Path("results/px4-autonomy-transit-v12/sequence-4752/result.json")]
    cases = []
    for path in paths:
        calls = json.loads(path.read_text())["calls"]
        planners = [c for c in calls if c["role"] == "planner" and not c.get("error")]
        if "occluded" in path.parent.name:
            cases.extend((path.parent.name, c) for c in planners[-4:])
        else:
            cases.extend((path.parent.name, c) for c in planners if c["choice"] in ("inspect", "dock"))
            cases.extend((path.parent.name, c) for c in planners if c["choice"].startswith("transit_"))
    gateway = JevGateway("openrouter", load_credential("openrouter", Path(".env")),
                         journal=args.out/"calls.jsonl")
    rows = []
    try:
        for trial, call in cases:
            state, question = call["state"], call["questions"]["selection"]
            for repeat in range(2):
                baseline = gateway.choose(state, question["instructions"], question["criteria"])
                mode = gateway.choose(state, INSTRUCTIONS, MODES)
                task = None
                if not mode.get("error"):
                    selected = mode["choice"]
                    criteria = {k:v for k,v in question["criteria"].items() if eligible(k,selected)}
                    task = gateway.choose({**state,"selected_task_kind":selected}, question["instructions"]+"\nYou selected task kind "+selected+". Select the concrete task now, using current evidence. All controls will be chosen separately by Jev.",criteria)
                row = dict(trial=trial,time=call["time"],repeat=repeat,recorded=call["choice"],
                    baseline=baseline.get("choice"),mode=mode.get("choice"),task=task.get("choice") if task else None,
                    errors=[r.get("error") for r in [baseline,mode,task or {}] if r.get("error")],
                    cost_usd=sum(r.get("usage",{}).get("cost",0) or 0 for r in [baseline,mode,task or {}]))
                rows.append(row);print(json.dumps(row),flush=True)
    finally:
        gateway.close()
        (args.out/"results.json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__ == "__main__":
    main()
