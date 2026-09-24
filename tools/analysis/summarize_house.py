"""Audit the eight-room flights without pooling them with earlier benchmarks."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from jev_drone.audit.native import audit


def main():
    output = Path("report/px4-house")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path("results").glob("px4-house-*/*/result.json")):
        row = audit(path)
        result = json.loads(path.read_text())
        calls = [
            json.loads(line) for line in path.with_name("calls.jsonl").read_text().splitlines()
        ]
        room_visits = []
        for call in calls:
            room = call["state"]["sensors"]["room"]
            if not room_visits or room_visits[-1] != room:
                room_visits.append(room)
        trace = result["trace"]
        positions = np.array([t["position"] for t in trace])
        row.update(
            course="course" in result,
            seed=result["world"]["config"]["seed"],
            room_visits=room_visits,
            distinct_rooms=len(set(room_visits)),
            path_metres=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            if len(positions) > 1
            else 0.0,
            task_outcomes=dict(Counter(t["outcome"] for t in result["tasks"])),
            request_errors=dict(Counter(c["error"] for c in calls if c.get("error"))),
            final_position=positions[-1].tolist() if len(positions) else None,
            valid_reports=[
                {"time": r["time"], "choice": r.get("choice", r.get("task"))}
                for r in result["reports"]
                if r["valid"]
            ],
        )
        rows.append(row)
    failures = []
    for path in sorted(Path("results").glob("px4-house-*/*/failure.txt")):
        if not path.with_name("result.json").exists():
            failures.append(
                {"directory": str(path.parent), "error": path.read_text().splitlines()[-1]}
            )
    probes = []
    for directory in sorted(Path("results").glob("px4-house-*-probe")):
        calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]
        probes.append(
            {
                "directory": str(directory),
                "requests": len(calls),
                "cost_usd": sum((c.get("usage") or {}).get("cost", 0) or 0 for c in calls),
                "errors": sum(c.get("http_status") != 200 or bool(c.get("error")) for c in calls),
            }
        )
    (output / "audit.json").write_text(
        json.dumps(
            {"flights": rows, "offline_probes": probes, "startup_or_process_failures": failures},
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Eight-room flight ledger",
        "",
        "Strict pass requires all objectives and return, no contacts, movement-rule violations, false reports, or audit problems.",
        "",
        "| Cohort / trial | Outcome | Stages | Strict | Contacts | Rule violations | Sim / wall seconds | Rooms visited |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| [{r['cohort']}/{r['id']}](../../{r['directory']}/result.json) | {r['status']} | {r['stages']}/{r['mission_stages']} | {'pass' if r['strict_pass'] else 'fail'} | {r['contacts']} | {r['rule_violations']} | {r['sim_seconds']:.1f} / {r['wall_seconds']:.1f} | {' → '.join(r['room_visits'])} |"
        )
    if failures:
        lines += ["", "Preserved process failures:"] + [
            f"- [{r['directory']}](../../{r['directory']}/failure.txt): {r['error']}"
            for r in failures
        ]
    lines += [
        "",
        f"{len(rows)} completed flights; reported native API cost ${sum(r['cost_usd'] for r in rows):.4f}.",
        "Courses and reused-seed development retries are not fresh mission reliability evidence.",
        "",
        "## Decision timing",
        "",
        "Accepted controls per wall second includes planner time, sensing and stale-response rejection. It is not the MAVLink publisher frequency. Software-rendered simulations shared this host; wall time is not a controlled throughput benchmark.",
        "",
        "| Cohort / trial | Accepted control Hz | Local API p50 / p95 ms | Sim / wall factor |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cohort']}/{r['id']} | {r['accepted_controls_per_wall_second']:.2f} | {r['model_local_p50_ms']:.0f} / {r['model_local_p95_ms']:.0f} | {r['realtime_factor']:.2f} |"
        )
    lines += [
        "",
        "## Offline decision probes",
        "",
        "Recorded-state and synthetic decisions are not additional native flights.",
        "",
        "| Probe | Requests | Reported cost | Errors |",
        "|---|---:|---:|---:|",
    ]
    for p in probes:
        lines.append(
            f"| [{Path(p['directory']).name}](../../{p['directory']}/results.json) | {p['requests']} | ${p['cost_usd']:.6f} | {p['errors']} |"
        )
    lines += [
        "",
        f"Total reported API cost: ${sum(r['cost_usd'] for r in rows) + sum(p['cost_usd'] for p in probes):.4f}.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "flights": len(rows),
                "missions": sum(not r["course"] for r in rows),
                "strict_missions": sum(r["strict_pass"] and not r["course"] for r in rows),
                "contacts": sum(r["contacts"] for r in rows),
                "rule_violations": sum(r["rule_violations"] for r in rows),
                "audit_problems": {
                    r["directory"]: r["audit_problems"] for r in rows if r["audit_problems"]
                },
                "cost_usd": sum(r["cost_usd"] for r in rows),
                "offline_requests": sum(p["requests"] for p in probes),
                "offline_cost_usd": sum(p["cost_usd"] for p in probes),
                "failures": failures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
