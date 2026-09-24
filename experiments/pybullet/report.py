"""Aggregate every finished trial, including failures and infrastructure errors."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def summarize(result):
    calls = result["calls"]
    controls = [c for c in calls if c["role"] == "control"]
    accepted = [c for c in controls if c.get("accepted")]
    latency = [c["latency_seconds"] for c in controls]
    complete = result["status"] == "success"
    return {
        "id": result["id"],
        "case": result["world"]["config"]["case"],
        "rig": result["world"]["config"]["rig"],
        "memory": result["world"]["config"]["memory"],
        "status": result["status"],
        "stage": result["stage"],
        "strict_pass": complete
        and not result["violations"]
        and not result["contacts"]
        and all(r["valid"] for r in result["reports"]),
        "sim_seconds": result["sim_seconds"],
        "wall_seconds": result["wall_seconds"],
        "calls": len(calls),
        "controls": len(controls),
        "accepted_controls": len(accepted),
        "median_latency_ms": float(np.median(latency) * 1000) if latency else None,
        "p95_latency_ms": float(np.quantile(latency, 0.95) * 1000) if latency else None,
        "local_hz": len(controls) / sum(max(0.2, x) for x in latency) if latency else 0,
        "local_cycle_seconds": sum(max(0.2, x) for x in latency),
        "wall_mission_hz": len(accepted) / result["wall_seconds"],
        "cost_usd": result["cost_usd"],
        "violations": len(result["violations"]),
        "violation_types": dict(Counter(x for v in result["violations"] for x in v["reasons"])),
        "contacts": len(result["contacts"]),
        "near_miss_seconds": result["near_miss_seconds"],
        "min_clearance": min(0.5, result["min_clearance"]),
        "clearance_censored_at_0_5m": result["min_clearance"] >= 0.5,
        "stale": result["stale_responses"],
        "errors": dict(Counter(c["error"] for c in calls if c.get("error"))),
        "unpriced_errors": sum(bool(c.get("error")) and not c.get("usage") for c in calls),
        "models": dict(Counter(c["model"] for c in calls if c.get("model"))),
        "stalled_tasks": sum(t["outcome"] == "stalled" for t in result["tasks"]),
        "actions": dict(Counter(c.get("choice") for c in accepted)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    failures = []
    for directory in args.directories:
        for path in sorted(directory.glob("*.json")):
            if path.name in ("manifest.json", "protocol.json"):
                continue
            result = json.loads(path.read_text())
            if "calls" not in result:
                failures.append(result)
                continue
            rows.append({"path": str(path), **summarize(result)})
    summary = {
        "episodes": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "strict_passes": sum(r["strict_pass"] for r in rows),
        "calls": sum(r["calls"] for r in rows),
        "cost_usd": sum(r["cost_usd"] for r in rows),
        "infrastructure_failures": failures,
        "rows": rows,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    if rows:
        with args.out.with_suffix(".csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                for row in rows
            )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    for row in rows:
        print(
            f"{row['id']}: {row['status']}, stage={row['stage']}, violations={row['violations']}, {row['sim_seconds']:.1f}s, ${row['cost_usd']:.4f}"
        )


if __name__ == "__main__":
    main()
