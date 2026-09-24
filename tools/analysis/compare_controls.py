"""Audit a completed, preregistered paired control-interface experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.legacy.native_report import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, raw, protocols = [], [], []
    for directory in args.directories:
        protocol = json.loads((directory/"protocol.json").read_text())
        protocols.append(protocol)
        for trial in protocol["trials"]:
            name = f"{trial['case']}-{trial['seed']}-{trial['control_interface']}"
            path = directory/name/"result.json"
            rows.append({**audit(path), "seed": trial["seed"]})
            raw.append(json.loads(path.read_text()))
    aggregate = {}
    for mode in ("full", "patch"):
        selected = [r for r in rows if r["control_interface"] == mode]
        controls = [c for d in raw if d["world"]["config"]["control_interface"] == mode
                    for c in d["calls"] if c["role"] == "control"]
        wall = sum(r["wall_seconds"] for r in selected)
        accepted = sum(r["accepted_controls"] for r in selected)
        latency = [c["latency_seconds"] for c in controls]
        aggregate[mode] = {"trials": len(selected), "completions": sum(r["status"] == "success" for r in selected),
            "strict_passes": sum(r["strict_pass"] for r in selected),
            "contact_episodes": sum(r["contacts"] > 0 for r in selected),
            "rule_violations": sum(r["rule_violations"] for r in selected),
            "total_wall_seconds": wall, "accepted_control_hz": accepted/wall,
            "accepted_controls": accepted, "local_requests": len(controls),
            "control_confirmations": sum(r["unchanged_control_confirmations"] for r in selected),
            "keep_choices": sum(r["keep_choices"] for r in selected),
            "assigned_channels": sum(r["assigned_velocity_channels"] for r in selected),
            "channels_per_accepted_control": sum(r["assigned_velocity_channels"] for r in selected)/accepted,
            "published_vector_changes": sum(r["published_vector_changes"] for r in selected),
            "published_vector_changes_per_second": sum(r["published_vector_changes"] for r in selected)/wall,
            "api_p50_ms": float(np.median(latency)*1000), "api_p95_ms": float(np.quantile(latency, .95)*1000),
            "revision_rejections": sum(r["revision_rejections"] for r in selected),
            "moving_lease_expirations": sum(r["moving_lease_expirations"] for r in selected),
            "cost_usd": sum(r["cost_usd"] for r in selected),
            "mean_local_input_tokens": float(np.mean([c["usage"]["input_tokens"] for c in controls
                                                       if "input_tokens" in c.get("usage", {})]))}
    pairs = []
    for row in rows:
        if row["control_interface"] != "full":
            continue
        patch = next(r for r in rows if (r["case"], r["seed"], r["control_interface"]) == (row["case"], row["seed"], "patch"))
        pairs.append({"case": row["case"], "seed": row["seed"],
                      "full": {k: row[k] for k in ("status", "strict_pass", "wall_seconds", "rule_violations")},
                      "patch": {k: patch[k] for k in ("status", "strict_pass", "wall_seconds", "rule_violations")},
                      "patch_minus_full_seconds": (patch["wall_seconds"]-row["wall_seconds"]
                          if patch["status"] == row["status"] == "success" else None)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"audit.json").write_text(json.dumps(rows, indent=2))
    (args.out/"comparison.json").write_text(json.dumps({"protocols": protocols, "aggregate": aggregate, "pairs": pairs}, indent=2))
    table = ["| Scene | Seed | Full commands | Partial updates | Difference |",
             "|---|---:|---|---|---:|"]
    for pair in pairs:
        def cell(mode):
            r = pair[mode]
            return f"{r['status']}, {r['wall_seconds']:.1f}s, {r['rule_violations']} rule violations"
        difference = f"{pair['patch_minus_full_seconds']:+.1f}s" if pair["patch_minus_full_seconds"] is not None else "not comparable"
        table.append(f"| {pair['case']} | {pair['seed']} | {cell('full')} | {cell('patch')} | {difference} |")
    (args.out/"pairs.md").write_text("\n".join(table)+"\n")
    print(json.dumps({"aggregate": aggregate, "audit_problems": {r["id"]: r["audit_problems"] for r in rows if r["audit_problems"]}}, indent=2))


if __name__ == "__main__":
    main()
