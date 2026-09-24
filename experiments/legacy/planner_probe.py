"""Replay recorded task decisions; compare transport modes without simulating flight."""

import argparse
import json
from pathlib import Path

from experiments.legacy.hierarchy import ask
from experiments.legacy.policy import Jev, load_key
from experiments.legacy.search import deepseek_ask
from experiments.legacy.task_planning import TASK_INSTRUCTIONS, task_options
from jev_drone.evidence import snapshot_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["jev", "low", "none"], required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    samples = []
    selections = [
        ("architecture-screen-task-jev-occluded", 0),
        ("architecture-screen-task-jev-near", 1),
        ("architecture-screen-task-jev-near", 2),
        ("architecture-screen-task-jev-absent", -1),
    ]
    for run, index in selections:
        e = json.loads((Path("results") / run / "episode.json").read_text())
        plans = [c for c in e["calls"] if c["role"] == "planner"]
        p = plans[index]
        samples.append(
            {"source": run, "index": index, "state": p["state"], "expected": p["choice"]}
        )
    client = Jev(load_key(args.key_file))
    results = []
    try:
        for sample in samples:
            memory = {o["id"]: o for o in sample["state"]["observed_objects"]}
            _, criteria = task_options(memory)
            if args.mode == "jev":
                result = ask(client.client, sample["state"], TASK_INSTRUCTIONS, criteria)
            else:
                result = deepseek_ask(
                    client.client,
                    sample["state"],
                    TASK_INSTRUCTIONS,
                    criteria,
                    max_tokens=4096,
                    reasoning_effort=args.mode,
                )
            results.append(
                {
                    **sample,
                    **result,
                    "agrees_with_recorded_choice": result.get("choice") == sample["expected"],
                }
            )
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "choice": result.get("choice"),
                        "expected": sample["expected"],
                        "latency": result["latency_seconds"],
                        "error": result.get("error"),
                    }
                ),
                flush=True,
            )
    finally:
        client.close()
        (args.out / "probe.json").write_text(
            json.dumps({"mode": args.mode, "results": results}, indent=2)
        )


if __name__ == "__main__":
    main()
