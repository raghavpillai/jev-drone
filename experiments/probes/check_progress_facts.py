"""Paired Jev decisions on recorded blocked states; never commands a vehicle."""

import argparse
import json
from pathlib import Path

from jev_drone.control.progress_facts import INSTRUCTIONS, enrich
from jev_drone.gateway import JevGateway, load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = [json.loads(line) for line in args.journal.read_text().splitlines()]
    blocked = [
        c
        for c in source
        if c["state"].get("task", {}).get("name") == "return_launch"
        and c["state"]["sensors"]["body_clearance_m"]["forward"] == 0
        and c.get("answers", {}).get("selection", {}).get("choice") == "keep_controls"
    ]
    selected = [blocked[int(i * (len(blocked) - 1) / 3)] for i in range(4)]
    args.out.mkdir(parents=True, exist_ok=False)
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for index, call in enumerate(selected):
            question = call["questions"]["selection"]
            for variant in ("baseline", "progress_facts"):
                state, criteria = call["state"], question["criteria"]
                instructions = question["instructions"]
                if variant == "progress_facts":
                    state, criteria = enrich(state, criteria)
                    instructions += INSTRUCTIONS
                result = gateway.choose(state, instructions, criteria)
                row = {
                    "index": index,
                    "variant": variant,
                    "choice": result.get("choice"),
                    "latency_seconds": result["latency_seconds"],
                    "error": result.get("error"),
                    "cost_usd": result.get("usage", {}).get("cost"),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        gateway.close()
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
