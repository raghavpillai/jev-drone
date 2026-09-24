"""Measure a lossless criteria deduplication against recorded control decisions."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential


def compact(criteria, state):
    result = {}
    for name in criteria:
        if name.startswith("detour_"):
            text = "Stop and select this bypass. Read its planned_route and route_obstacles."
        elif name == "brake_and_replan":
            text = "Stop all axes and return a blocked/failed task to the planner. Not ordinary braking or turning."
        elif name == "brake":
            text = "Stop translation AND pan."
        elif name == "keep_controls":
            text = "Keep ALL current axes, including ongoing pan."
        else:
            text = "Apply this stick-state intent; omitted channels retain their values."
        result[name] = (
            text
            + " Read control_effects."
            + name
            + " for exact patch, resulting_controls, movement_violations, travel_effect and turn_effect."
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    source = Path("results/px4-house-v29-transitions/house_hidden-5111/calls.jsonl")
    categories = {}
    with source.open() as stream:
        for line in stream:
            if '"control_effects"' not in line:
                continue
            c = json.loads(line)
            s = c["state"]
            choice = c.get("answers", {}).get("selection", {}).get("choice")
            category = (
                "detour"
                if choice and choice.startswith("detour_")
                else "pan"
                if choice and choice.startswith("pan_")
                else "vertical"
                if choice in ("down", "down_creep", "up", "up_creep")
                else "cruise"
                if choice == "forward_full"
                else "brake"
                if choice == "brake"
                else "strafe"
                if choice and choice.startswith("strafe_")
                else None
            )
            if category and category not in categories:
                categories[category] = c
            if len(categories) == 6:
                break
    (a.out / "protocol.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "categories": list(categories),
                "repeats": 2,
                "variants": ["original", "compact"],
                "scope": "Preserve every control option and all effect data in state; remove duplicated patch/route data from criteria only. Compare response latency, bytes, unsafe choices, and choice agreement. Not a native flight or reliability test.",
            },
            indent=2,
        )
        + "\n"
    )
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=a.out / "calls.jsonl"
    )
    rows = []
    try:
        for category, call in categories.items():
            for repeat in range(2):
                for variant in ("original", "compact") if repeat == 0 else ("compact", "original"):
                    s = deepcopy(call["state"])
                    q = deepcopy(call["questions"]["selection"])
                    if variant == "compact":
                        q["criteria"] = compact(q["criteria"], s)
                    response = gateway.choose(s, q["instructions"], q["criteria"])
                    choice = response.get("choice")
                    row = dict(
                        category=category,
                        repeat=repeat,
                        variant=variant,
                        choice=choice,
                        error=response.get("error"),
                        latency=response["latency_seconds"],
                        request_bytes=len(json.dumps([s, q])),
                        violations=s["control_effects"].get(choice, {}).get("movement_violations"),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (a.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
