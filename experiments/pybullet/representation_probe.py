"""Live Jev counterfactuals on fixed dangerous observations; no simulated success claims."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.world.world import DIRECTIONS


def action_facts(state):
    sensor = state["sensors"]
    velocity = np.array(sensor["velocity_estimate"])
    moving = np.linalg.norm(velocity) > .15
    facts = {}
    for name, direction in DIRECTIONS.items():
        clearance = sensor["clearance_m"][name]
        axis_ok = not moving or np.dot(velocity/np.linalg.norm(velocity), direction) >= .95
        facts[name] = {"clearance": clearance,
                       "depth_label": "UNKNOWN" if clearance is None else "OPEN" if clearance >= 1.3 else "SLOW_ONLY" if clearance >= .4 else "BLOCKED",
                       "axis_change_ready": bool(axis_ok),
                       "cruise_clearance_sufficient": bool(clearance is not None and clearance >= 1.3),
                       "slow_clearance_sufficient": bool(clearance is not None and clearance >= .4)}
    return facts


def modify(state, method):
    state = deepcopy(state)
    facts = action_facts(state)
    if method == "labels":
        state["sensors"]["clearance_m"] = {k: {"meters": v["clearance"], "status": v["depth_label"]} for k, v in facts.items()}
    elif method == "action_facts":
        state["movement_checks"] = facts
    elif method == "compact":
        state = {"goal_offset": state["goal_offset"], "at_position": state["at_position"],
                 "motion_direction": state.get("motion_direction"), "motion_axis_needs_travel": state.get("motion_axis_needs_travel"),
                 "stopped": state.get("stopped"), "movement_checks": facts,
                 "instruction": "BLOCKED or UNKNOWN: do not translate there. SLOW_ONLY: brake then slow; never cruise. Brake before axis change. Choose camera or brake when appropriate."}
    return state


def valid(choice, original):
    if not choice or not choice.startswith(("cruise_", "slow_")):
        return choice is not None
    speed, direction = choice.split("_", 1)
    facts = action_facts(original)[direction]
    return facts["axis_change_ready"] and facts["cruise_clearance_sufficient" if speed == "cruise" else "slow_clearance_sufficient"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = []
    for path in args.episodes:
        result = json.loads(path.read_text())
        controls = [c for c in result["calls"] if c["role"] == "control" and c.get("accepted") and not valid(c.get("choice"), c["state"])]
        for index, call in enumerate(controls[:6]):
            for method in ("raw", "labels", "action_facts", "compact"):
                jobs.append((path.stem, index, method, call))
    (args.out/"protocol.json").write_text(json.dumps({"input_episodes": [str(p) for p in args.episodes], "methods": ["raw", "labels", "action_facts", "compact"], "count": len(jobs)}, indent=2))
    def run(job):
        episode, index, method, call = job
        gateway = JevGateway("openrouter", load_credential("openrouter", args.key_file))
        try:
            result = gateway.choose(modify(call["state"], method), call["questions"]["selection"]["instructions"], call["questions"]["selection"]["criteria"])
        finally:
            gateway.close()
        row = {"episode": episode, "index": index, "method": method, "valid": valid(result.get("choice"), call["state"]), "original_choice": call.get("choice"), **result}
        (args.out/f"{episode}-{index}-{method}.json").write_text(json.dumps(row))
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, jobs))
    summary = {method: {"valid": sum(r["valid"] for r in rows if r["method"] == method),
                        "count": sum(r["method"] == method for r in rows),
                        "choices": [r.get("choice", r.get("error")) for r in rows if r["method"] == method]}
               for method in ("raw", "labels", "action_facts", "compact")}
    summary["cost_usd"] = sum(r.get("usage", {}).get("cost", 0) or 0 for r in rows)
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
