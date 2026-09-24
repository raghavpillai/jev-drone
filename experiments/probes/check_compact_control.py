"""Paired live Jev probes of shorter instructions on stalled and nominal states."""

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.representations.compact_control import INSTRUCTIONS
from jev_drone.gateway import JevGateway, load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for path in sorted(Path("results/px4-autonomy-modes-validation").glob("*/result.json")):
        result = json.loads(path.read_text())
        stationary = []
        for call in result["calls"]:
            state = call["state"]
            if (
                call["role"] == "control"
                and call.get("accepted")
                and call["choice"] in ("brake", "keep_controls")
                and not state["at_position"]
                and state["sensors"]["speed"] < 0.07
                and abs(state["sensors"]["yaw_rate_rps"]) < 0.07
                and not state["mission_progress"]["target_in_inspection_range"]
                and not state["mission_progress"]["dock_in_range"]
            ):
                if not stationary or call["time"] - stationary[-1]["time"] > 15:
                    stationary.append(call)
        cases += [(path.parent.name, "stationary", c) for c in stationary[:2]]
    result = json.loads(
        Path("results/px4-recovery-depth-v15/controller-4910/result.json").read_text()
    )
    for name in (
        "forward_slow",
        "backward_slow",
        "strafe_left_slow",
        "strafe_right_slow",
        "pan_left",
        "pan_right",
        "up",
        "down",
    ):
        call = next(
            (
                c
                for c in result["calls"]
                if c["role"] == "control" and c.get("accepted") and c["choice"] == name
            ),
            None,
        )
        if call:
            cases.append((result["id"], "nominal", call))
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for trial, kind, call in cases:
            question = call["questions"]["selection"]
            for variant in ("baseline", "compact"):
                instructions = question["instructions"] if variant == "baseline" else INSTRUCTIONS
                result = gateway.choose(call["state"], instructions, question["criteria"])
                effect = call["state"]["control_effects"].get(result.get("choice"), {})
                row = {
                    "trial": trial,
                    "kind": kind,
                    "time": call["time"],
                    "variant": variant,
                    "choice": result.get("choice"),
                    "violations": effect.get("movement_violations"),
                    "nonzero": any(effect.get("resulting_controls", {}).values()),
                    "detour": result.get("choice", "").startswith("detour_"),
                    "latency_seconds": result["latency_seconds"],
                    "error": result.get("error"),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(Counter((r["variant"], r["kind"], r["nonzero"], bool(r["violations"])) for r in rows))


if __name__ == "__main__":
    main()
