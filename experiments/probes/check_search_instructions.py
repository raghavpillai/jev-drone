"""Compare Jev search decisions at a recorded premature give-up state."""

import argparse
import json
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.search_memory import INSTRUCTIONS, SearchMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path("results/px4-autonomy-braking-v10/occluded-4753/result.json")
    calls = json.loads(source.read_text())["calls"]
    memory = SearchMemory()
    for call in calls:
        memory.observe(call["state"]["sensors"], call["time"])
    call = calls[-1]
    assert call["choice"] == "give_up" and call["role"] == "planner"
    gateway = JevGateway(
        "openrouter", load_credential("openrouter", Path(".env")), journal=args.out / "calls.jsonl"
    )
    rows = []
    try:
        for repeat in range(4):
            for variant in ("baseline", "search_memory"):
                state = call["state"]
                question = call["questions"]["selection"]
                instructions, criteria = question["instructions"], question["criteria"]
                if variant == "search_memory":
                    state, criteria = memory.describe(state, criteria)
                    instructions += INSTRUCTIONS
                result = gateway.choose(state, instructions, criteria)
                row = dict(
                    repeat=repeat,
                    variant=variant,
                    choice=result.get("choice"),
                    error=result.get("error"),
                    latency_seconds=result["latency_seconds"],
                    cost_usd=result.get("usage", {}).get("cost", 0),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
