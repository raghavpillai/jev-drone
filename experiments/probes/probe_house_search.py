"""Paired recorded-state Jev search probes; these never fly the drone."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.search_memory import INSTRUCTIONS, SearchMemory
from jev_drone.world.layout import HOUSE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    path = Path("results/px4-house-v29-transitions/house_hidden-5111")
    result = json.loads((path / "result.json").read_text())
    raw = [json.loads(x) for x in (path / "calls.jsonl").read_text().splitlines()]
    memory = SearchMemory(HOUSE)
    # Freeze old candidate positions for all arms: this isolates information,
    # not the separate lower-height intervention being tested in native flight.
    for name in list(memory.viewpoints):
        if name.endswith("_2.2"):
            view = memory.viewpoints.pop(name)
            view["position"][2] = 2.5
            memory.viewpoints[name[:-3] + "2.5"] = view
    cases = []
    controls = set()
    for log, call in zip(result["calls"], raw):
        s = call["state"]
        memory.observe(s["sensors"], log["time"])
        role = log["role"]
        kind = s.get("selected_task_kind")
        if role == "planner" and kind == "search":
            cases.append(("search_" + str(round(log["time"])), call, deepcopy(memory)))
        if role == "planner_mode" and log["time"] < 640:
            label = (
                "outside"
                if s["sensors"]["room"] == "library"
                else "known_target"
                if s.get("target_visible") and not s.get("inspection_ready")
                else "ready"
                if s.get("inspection_ready")
                else None
            )
            if label and label not in controls:
                controls.add(label)
                cases.append((label, call, deepcopy(memory)))
    protocol = {
        "source": str(path),
        "variants": ["baseline", "instructions", "survey_facts"],
        "repeats": 2,
        "cases": [c[0] for c in cases],
        "interpretation": "Recorded-state choices only. Compare search task/height selection and preservation of outside/change_room, known_target/approach, ready/report. No target coordinates beyond original observed state; no flight-success claims. All arms retain identical offered choice names/positions. Full facts add inward look semantics and settled camera coverage; native flight separately tests lower heights.",
    }
    (a.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=a.out / "calls.jsonl"
    )
    rows = []
    try:
        for label, call, mem in cases:
            for repeat in range(2):
                variants = (
                    protocol["variants"] if repeat == 0 else list(reversed(protocol["variants"]))
                )
                for variant in variants:
                    state = deepcopy(call["state"])
                    q = deepcopy(call["questions"]["selection"])
                    if variant != "baseline":
                        q["instructions"] += INSTRUCTIONS
                    if variant == "survey_facts":
                        original = dict(q["criteria"])
                        q["criteria"].setdefault("give_up", "No useful task remains.")
                        state, q["criteria"] = mem.describe(state, q["criteria"])
                        q["criteria"] = {k: v for k, v in q["criteria"].items() if k in original}
                        for name, option in state.get("options", {}).items():
                            if name.startswith("view_"):
                                lo, hi = HOUSE.rooms[state["sensors"]["room"]]
                                option["look_position"] = [
                                    (lo[0] + hi[0]) / 2,
                                    (lo[1] + hi[1]) / 2,
                                    option["position"][2],
                                ]
                                option["purpose"] = "Reach this position THEN look into the room."
                    response = gateway.choose(state, q["instructions"], q["criteria"])
                    choice = response.get("choice")
                    option = state.get("options", {}).get(choice, {})
                    row = dict(
                        case=label,
                        repeat=repeat,
                        variant=variant,
                        choice=choice,
                        selected_position=option.get("position"),
                        error=response.get("error"),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (a.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
