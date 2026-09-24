"""Recorded-state Jev comparison; this candidate is not the live flight policy."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential


CONTINUITY = """
For return_to_launch, a retrace waypoint that ARRIVED is evidence that the return
strategy is working. If earlier retrace options remain, continue that strategy
across rooms instead of abandoning it after one waypoint. A failed direct door
approach remains failed after a small retrace step; changing task family does not
remove the obstruction. Reconsider retracing if its own waypoint stalls or needs
replanning. Do not override observed occupied endpoints or a ready docking report.
For outbound inspection/search, this return rule does not apply. You choose the
task kind; this instruction never executes or forces a movement.
"""


def states():
    # Recorded failures and controls selected before making any probe request.
    sources = [
        ("px4-recovery-endpoints-v22/occluded-4872", "after_retrace", 2),
        ("px4-recovery-precision-v21/occluded-4872", "after_retrace", 1),
        ("px4-recovery-precision-v21/blocked_door-4875", "after_retrace", 2),
        ("px4-recovery-endpoints-v22/occluded-4872", "outbound", 2),
        ("px4-recovery-return-v16/sequence-4873", "dock_ready", 1),
    ]
    selected = []
    for name, kind, count in sources:
        root = Path("results")/name
        result = json.loads((root/"result.json").read_text())
        journal = [json.loads(line) for line in (root/"calls.jsonl").read_text().splitlines()]
        found = []
        for index, (call, raw) in enumerate(zip(result["calls"], journal)):
            if call["role"] != "planner_mode" or raw.get("error"):
                continue
            state = raw["state"]
            previous = state.get("recent_tasks", [])[-1:]
            match = (previous and previous[0]["name"].startswith("retrace_")
                     and previous[0]["outcome"] == "arrived"
                     and any(n.startswith("retrace_") for n in state["options"])) if kind == "after_retrace" else (
                state["dock_ready"] if kind == "dock_ready" else "marker" in state["objective"])
            if match:
                found.append((name, kind, index, call["time"], raw))
        if len(found) < count:
            raise ValueError(f"Only {len(found)} states available for {name}/{kind}")
        selected.extend(found[:count])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = states()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"protocol.json").write_text(json.dumps({
        "states": [{"trial": n, "kind": k, "call_index": i, "sim_time": t} for n,k,i,t,_ in selected],
        "repeats": 2, "variants": ["baseline", "return_continuity"],
        "candidate_instruction": CONTINUITY,
        "limitation": "Mode selection probe, not a flight success or route safety test.",
    }, indent=2)+"\n")
    gateway = JevGateway("openrouter", load_credential("openrouter", Path(".env")),
                         journal=args.out/"calls.jsonl")
    rows = []
    try:
        for name, kind, index, time, raw in selected:
            for repeat in range(2):
                variants = ["baseline", "return_continuity"]
                if repeat:
                    variants.reverse()
                for variant in variants:
                    question = deepcopy(raw["questions"]["selection"])
                    if variant != "baseline":
                        question["instructions"] += CONTINUITY
                    response = gateway.choose(raw["state"], question["instructions"], question["criteria"])
                    row = dict(trial=name, kind=kind, call_index=index, sim_time=time, repeat=repeat,
                               variant=variant, choice=response.get("choice"), error=response.get("error"),
                               latency_seconds=response["latency_seconds"])
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out/"results.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
