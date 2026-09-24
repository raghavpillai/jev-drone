"""Alternate identical Jev requests across gateways; keep cold calls and failures."""

import argparse
import json
import statistics
import time
from pathlib import Path

from experiments.legacy.policy import CRITERIA, INSTRUCTIONS, INTENT_CRITERIA, INTENT_INSTRUCTIONS
from experiments.legacy.representation_experiment import GUIDANCE
from jev_drone.evidence import snapshot_sources
from jev_drone.gateway import JevGateway, load_credential


def requests():
    path = Path("results") / "representation-brief-screen/brief-overpass-clean.json"
    decisions = json.loads(path.read_text())["decisions"]
    questions = {
        "movement": {
            "type": "choice",
            "instructions": INSTRUCTIONS + GUIDANCE,
            "criteria": {k: v for k, v in CRITERIA.items() if "_" not in k},
        },
        "intent": {
            "type": "choice",
            "instructions": INTENT_INSTRUCTIONS + GUIDANCE,
            "criteria": INTENT_CRITERIA,
        },
    }
    return [(decisions[index]["state"], questions) for index in (0, 8, 20, 40, 60)]


def summarize(calls):
    summary = {}
    for provider in sorted({c["provider"] for c in calls}):
        measured = [c for c in calls if c["provider"] == provider and not c["warmup"]]
        valid = [c for c in measured if not c.get("error")]
        latency = sorted(c["latency_seconds"] for c in valid)
        cold = next((c for c in calls if c["provider"] == provider), None)
        summary[provider] = {
            "measured_requests": len(measured),
            "valid": len(valid),
            "errors": len(measured) - len(valid),
            "over_800ms": sum(c["latency_seconds"] > 0.8 for c in measured),
            "cold_ms": 1000 * cold["latency_seconds"] if cold else None,
            "median_ms": 1000 * statistics.median(latency) if latency else None,
            "p95_ms": 1000 * latency[min(len(latency) - 1, int(0.95 * len(latency)))]
            if latency
            else None,
            "serial_requests_per_second": len(measured)
            / sum(c["latency_seconds"] for c in measured)
            if measured
            else None,
            "reported_cost_usd": sum(c.get("usage", {}).get("cost", 0) or 0 for c in measured),
            "returned_models": sorted({c["model"] for c in valid if c.get("model")}),
        }
    pairs = {}
    for c in calls:
        if not c["warmup"] and not c.get("error"):
            pairs.setdefault(c["round"], {})[c["provider"]] = c["answers"]["movement"]["choice"]
    complete = [p for p in pairs.values() if len(p) == 2]
    return {
        "providers": summary,
        "paired_valid": len(complete),
        "movement_agreement": sum(len(set(p.values())) == 1 for p in complete) / len(complete)
        if complete
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers", nargs="+", choices=["openrouter", "vercel"], default=["openrouter", "vercel"]
    )
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gateways, unavailable = {}, {}
    for provider in args.providers:
        try:
            gateways[provider] = JevGateway(provider, load_credential(provider, args.key_file))
        except ValueError:
            unavailable[provider] = "missing_credentials"
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    calls = []
    examples = requests()
    try:
        for i in range(args.rounds + 2):
            order = list(gateways)
            if i % 2:
                order.reverse()
            state, questions = examples[i % len(examples)]
            for provider in order:
                result = gateways[provider].evaluate(state, questions)
                calls.append({"round": i - 2, "warmup": i < 2, "unix_time": time.time(), **result})
                (args.out / "benchmark.json").write_text(
                    json.dumps(
                        {"unavailable": unavailable, "calls": calls, **summarize(calls)}, indent=2
                    )
                )
    finally:
        for gateway in gateways.values():
            gateway.close()
    result = {"unavailable": unavailable, "calls": calls, **summarize(calls)}
    (args.out / "benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "calls"}, indent=2))


if __name__ == "__main__":
    main()
