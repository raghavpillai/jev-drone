"""Regenerate the recovery experiment ledger from preserved native trials."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from jev_drone.audit.native import audit


def summarize(path):
    row = audit(path)
    result = json.loads(path.read_text())
    calls = [json.loads(line) for line in path.with_name("calls.jsonl").read_text().splitlines()]
    row.update(
        course="course" in result,
        seed=result["world"]["config"]["seed"],
        request_errors=dict(Counter(c["error"] for c in calls if c.get("error"))),
        http_responses=sum("http_status" in c for c in calls),
        valid_model_responses=sum("answers" in c and not c.get("error") for c in calls),
        valid_reports=[{"choice": r.get("choice", r.get("task")), "stage": r.get("stage"), "time": r["time"]}
                       for r in result["reports"] if r["valid"]],
        task_outcomes=dict(Counter(t["outcome"] for t in result["tasks"])),
        final_estimated_position=calls[-1]["state"]["sensors"]["position_estimate"] if calls else None,
    )
    manifest = json.loads(path.with_name("manifest.json").read_text())
    row["snapshot_hash"] = hashlib.sha256(
        json.dumps(manifest["sha256"], sort_keys=True).encode()).hexdigest()
    return row


def main():
    output = Path("report/px4-recovery")
    output.mkdir(parents=True, exist_ok=True)
    rows = [summarize(p) for p in sorted(Path("results").glob("px4-recovery-*/*/result.json"))]
    cohorts = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        batch = [r for r in rows if r["cohort"] == cohort]
        cohorts[cohort] = {
            "trials": len(batch), "strict_passes": sum(r["strict_pass"] for r in batch),
            "mission_trials": sum(not r["course"] for r in batch),
            "mission_strict_passes": sum(r["strict_pass"] and not r["course"] for r in batch),
            "contact_trials": sum(bool(r["contacts"]) for r in batch),
            "rule_violation_trials": sum(bool(r["rule_violations"]) for r in batch),
            "audit_problem_trials": sum(bool(r["audit_problems"]) for r in batch),
            "request_attempts": sum(r["calls"] for r in batch),
            "reported_cost_usd": sum(r["cost_usd"] for r in batch),
            "source_snapshots": sorted({r["snapshot_hash"] for r in batch}),
        }
    offline = []
    for path in sorted(Path("results").glob("px4-recovery-*-offline/calls.jsonl")):
        calls = [json.loads(line) for line in path.read_text().splitlines()]
        offline.append({"experiment": path.parent.name, "attempts": len(calls),
                        "valid_responses": sum("answers" in c and not c.get("error") for c in calls),
                        "reported_cost_usd": sum(float(c.get("usage", {}).get("cost", 0)) for c in calls)})
    failed_attempts = []
    for path in sorted(Path("results").glob("px4-recovery-*/*/failure.txt")):
        if path.with_name("result.json").exists():
            continue
        config = json.loads(path.with_name("config.json").read_text())
        journal = path.with_name("calls.jsonl")
        calls = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        manifest = json.loads(path.with_name("manifest.json").read_text())
        changed = [name for name, digest in manifest["sha256"].items()
                   if hashlib.sha256((path.parent/"source"/name).read_bytes()).hexdigest() != digest]
        failed_attempts.append({"directory": str(path.parent), "case": config["case"], "seed": config["seed"],
                                "error": path.read_text().strip().splitlines()[-1], "request_attempts": len(calls),
                                "changed_source_files": changed,
                                "reported_cost_usd": sum(float(c.get("usage", {}).get("cost", 0)) for c in calls)})
    summary = {
        "native_trials": len(rows), "native_request_attempts": sum(r["calls"] for r in rows),
        "native_valid_responses": sum(r["valid_model_responses"] for r in rows),
        "native_reported_cost_usd": sum(r["cost_usd"] for r in rows),
        "offline": offline, "cohorts": cohorts, "failed_attempts_without_result": failed_attempts,
        "notes": ["Development seeds, courses, and fresh validation are separate populations.",
                  "Request attempts include local serialization errors that never reached the API.",
                  "Reported costs exclude any unreported charges on failed requests.",
                  "Source snapshot hash covers all frozen package files, including tests and protocols."],
    }
    (output/"audit.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    lines = ["# Native recovery experiment ledger", "",
             "Generated with `python summarize_px4_recovery.py`. All model choices use OpenRouter Jev.",
             "Strict completion requires every objective and return, with zero contacts, movement-rule violations, false reports, and audit problems.",
             "Request attempts include client-side errors. Development trials are not a reliability estimate.", ""]
    for cohort in cohorts:
        lines += ["## "+cohort, "",
                  "| Trial | Result | Stages | Strict | Contacts | Rule violations | Sim / wall s | Accepted control Hz | Attempts | Cost |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for r in (r for r in rows if r["cohort"] == cohort):
            label = Path(r["directory"]).name+(" (course)" if r["course"] else "")
            link = "../../"+r["directory"]+"/result.json"
            lines.append(f"| [{label}]({link}) | {r['status']} | {r['stages']}/{r['mission_stages']} | "
                         f"{'pass' if r['strict_pass'] else 'fail'} | {r['contacts']} | {r['rule_violations']} | "
                         f"{r['sim_seconds']:.1f} / {r['wall_seconds']:.1f} | "
                         f"{r['accepted_controls_per_wall_second']:.2f} | {r['calls']} | ${r['cost_usd']:.3f} |")
        lines.append("")
    if failed_attempts:
        lines += ["## Exceptions without a completed flight result", "",
                  "These attempts remain in the ledger. A startup failure is not a completed mission trial.", ""]
        for failure in failed_attempts:
            lines.append(f"- [{failure['directory']}](../../{failure['directory']}/failure.txt): "
                         f"{failure['error']} ({failure['request_attempts']} model request attempts).")
        lines.append("")
    total_cost = summary["native_reported_cost_usd"]+sum(r["reported_cost_usd"] for r in offline+failed_attempts)
    lines += [f"Native: {len(rows)} trials, {summary['native_request_attempts']} request attempts; "
              f"offline: {sum(r['attempts'] for r in offline)} attempts; "
              f"exceptions without flight results: {len(failed_attempts)}. Combined reported API cost: ${total_cost:.4f}.", "",
              "[Full audit and diagnostics](audit.json) · [Machine-readable summary](summary.json)", ""]
    (output/"results.md").write_text("\n".join(lines))
    print(json.dumps({"native_trials": len(rows), "cost_usd": total_cost,
                      "cohorts": {k: {n: v[n] for n in ("trials", "strict_passes", "audit_problem_trials")}
                                  for k, v in cohorts.items()}}, indent=2))


if __name__ == "__main__":
    main()
