"""Separate prospective full-mission validation from adaptive development."""

import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from jev_drone.audit.native import audit
from tools.analysis.analyze_doorways import analyze as analyze_doorways
from tools.analysis.analyze_house_search import analyze

COHORTS = {
    "px4-house-v34-full": "development reference",
    "px4-house-v35-return": "development return change",
    "px4-house-v36-validation": "v36 fresh-seed validation",
    "px4-house-v37-validation": "v37 fresh-seed validation",
    "px4-house-v38-validation": "v38 fresh-seed validation",
    "px4-house-v39-validation": "v39 fresh-seed validation",
    "px4-house-v36-paired-reference": "v36 paired reference",
    "px4-house-v40-scope": "v40 scoped follow-up",
}
PROBES = (
    "px4-house-v35-return-probe",
    "px4-house-v35-return-scope-probe",
    "px4-house-v36-axis-probe",
    "px4-house-v37-centering-probe",
    "px4-house-v37-pan-probe",
    "px4-house-v38-approach-probe",
    "px4-house-v39-presentation-probe",
    "px4-house-v39-violations-probe",
    "px4-house-v39-altitude-probe",
    "px4-house-v39-boundaries-probe",
    "px4-house-v39-uniform-probe",
    "px4-house-v39-native-order-probe",
    "px4-house-v40-height-probe",
    "px4-house-v40-scope-probe",
)


def main():
    output = Path("report/px4-full-missions")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    pending = []
    failures = []
    planned = Counter()
    for cohort, population in COHORTS.items():
        directory = Path("results") / cohort
        protocol = json.loads((directory / "protocol.json").read_text())
        for job in protocol["trials"]:
            trial_population = (
                ("v40 " + job["population"]) if cohort == "px4-house-v40-scope" else population
            )
            planned[trial_population] += 1
            trial = f"{job['case']}-{job['seed']}"
            path = directory / trial / "result.json"
            tag = cohort.removeprefix("px4-house-") + "--" + trial
            if not path.exists():
                failure = path.with_name("failure.txt")
                (failures if failure.exists() else pending).append(
                    {
                        "directory": str(path.parent),
                        "error": failure.read_text().splitlines()[-1] if failure.exists() else None,
                    }
                )
                continue
            folder = output / tag
            folder.mkdir(parents=True, exist_ok=True)
            row = audit(path)
            (folder / "audit.json").write_text(json.dumps(row, indent=2) + "\n")
            with redirect_stdout(io.StringIO()):
                analyze(path, folder)
            result = json.loads(path.read_text())
            phases = json.loads((folder / "analysis.json").read_text())
            journal = [
                json.loads(line) for line in path.with_name("calls.jsonl").read_text().splitlines()
            ]
            doorways = analyze_doorways(result, journal)
            (folder / "doorways.json").write_text(json.dumps(doorways, indent=2) + "\n")
            rooms = []
            return_rooms = []
            returning = False
            for call, raw in zip(result["calls"], journal):
                state = raw["state"]
                room = state["sensors"]["room"]
                objective = state.get(
                    "objective", state.get("mission_progress", {}).get("objective", {})
                )
                returning |= "dock" in objective
                if not rooms or rooms[-1] != room:
                    rooms.append(room)
                if returning and (not return_rooms or return_rooms[-1] != room):
                    return_rooms.append(room)
            return_tasks = [
                t
                for t in result["tasks"]
                if t.get("mission_stage") == len(result["world"]["mission"]) - 1
            ]
            families = Counter(t["name"].split("_")[0] for t in return_tasks)
            row.update(
                population=trial_population,
                tag=tag,
                room_visits=rooms,
                return_room_visits=return_rooms,
                first_seen=phases["first_target_seen"],
                inspected=phases["inspected"],
                phases=phases["phases"],
                return_task_families=dict(families),
                return_task_outcomes=dict(Counter(t["outcome"] for t in return_tasks)),
                doorways={k: v for k, v in doorways.items() if k != "tasks"},
                control_rejections=dict(
                    Counter(
                        c.get(
                            "rejection_reason",
                            "late_response"
                            if c["latency_seconds"]
                            > result["world"]["config"]["command_lease_seconds"]
                            else "error_or_mission_ended",
                        )
                        for c in result["calls"]
                        if c["role"] == "control" and not c.get("accepted")
                    )
                ),
            )
            rows.append(row)
    probes = []
    for cohort in PROBES:
        directory = Path("results") / cohort
        calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]
        probes.append(
            {
                "cohort": cohort,
                "requests": len(calls),
                "errors": sum(bool(c.get("error")) for c in calls),
                "cost_usd": sum((c.get("usage") or {}).get("cost", 0) or 0 for c in calls),
            }
        )
    summary = {
        "flights": rows,
        "probes": probes,
        "planned": dict(planned),
        "pending": pending,
        "process_failures": failures,
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Full-mission experiment ledger",
        "",
        "Strict pass requires all inspections and return to launch, no contacts, movement-rule violations or false reports, and a clean independent provenance audit. Development retries are kept separate from fresh-seed validation. All times are simulated seconds unless stated otherwise.",
        "",
        "| Trial | Population | Outcome | Objectives | Strict | Inspection time | Sim / wall seconds | Contacts / violations |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        inspected = "—" if r["inspected"] is None else f"{r['inspected']:.1f}"
        lines.append(
            f"| [{r['cohort']}/{r['id']}]({r['tag']}/phases.png) | {r['population']} | {r['status']} | {r['stages']}/{r['mission_stages']} | {'pass' if r['strict_pass'] else 'fail'} | {inspected} | {r['sim_seconds']:.1f} / {r['wall_seconds']:.1f} | {r['contacts']} / {r['rule_violations']} |"
        )
    validation = [r for r in rows if "validation" in r["population"]]
    lines += [""]
    for population, count in planned.items():
        if "validation" not in population:
            continue
        completed = [r for r in rows if r["population"] == population]
        lines.append(
            f"{population}: {sum(r['strict_pass'] for r in completed)}/{count} planned trials passed strictly; {len(completed)} completed records available."
        )
    lines += [
        "",
        "Versions are separate populations, not a pooled reliability estimate. Small samples on fixed furniture recipes do not establish broad reliability.",
        "",
        "## Return paths",
        "",
    ]
    for r in rows:
        route = " → ".join(r["return_room_visits"]) or "Return stage not reached"
        lines.append(
            f"- **{r['cohort']}/{r['id']}:** {route}. Task families: `{r['return_task_families']}`. Outcomes: `{r['return_task_outcomes']}`."
        )
    lines += [
        "",
        "## Doorway behavior",
        "",
        "These include outbound and return crossing tasks, but exclude ordinary approach tasks. Turning reversals include legitimate corrections; rates and scene routes differ across trials.",
        "",
        "| Trial | Crossing attempts | Arrived / stalled / replanned | Crossing seconds | Pan direction changes |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        d = r["doorways"]
        o = d["outcomes"]
        lines.append(
            f"| [{r['tag']}]({r['tag']}/doorways.json) | {d['attempts']} | {o.get('arrived', 0)} / {o.get('stalled', 0)} / {o.get('replan_requested', 0)} | {d['total_seconds']:.1f} | {d['pan_direction_changes']} |"
        )
    lines += [
        "",
        "## Timing and audit",
        "",
        "Accepted control Hz includes planning, sensing and expired-response rejection; it is not the setpoint publisher or renderer frequency. Up to five native worlds shared a 64-core host, with fixed per-trial wall watchdogs. API latency and wall throughput are not controlled provider benchmarks.",
        "",
        "| Trial | Accepted control Hz | API p50 / p95 ms | Rejected stale controls | Audit issues | Reported cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| [{r['tag']}]({r['tag']}/audit.json) | {r['accepted_controls_per_wall_second']:.2f} | {r['model_local_p50_ms']:.0f} / {r['model_local_p95_ms']:.0f} | {r['stale_responses']} | {len(r['audit_problems'])} | ${r['cost_usd']:.4f} |"
        )
    lines += [
        "",
        "Rejections include responses whose control-state revision changed during the request, even when API latency itself was below the lease. These safeguards were not relaxed.",
        "",
    ]
    for r in rows:
        lines.append(f"- **{r['tag']}:** `{r['control_rejections']}`.")
    lines += [
        "",
        "## Offline probes",
        "",
        "These are recorded-state choices, not additional flight successes.",
        "",
    ]
    for p in probes:
        lines.append(
            f"- [{p['cohort']}](../../results/{p['cohort']}/results.json): {p['requests']} requests, {p['errors']} errors, ${p['cost_usd']:.6f}."
        )
    if pending or failures:
        lines += ["", f"Pending: {pending}", f"Process failures: {failures}"]
    lines += [
        "",
        f"Reported total cost: ${sum(r['cost_usd'] for r in rows) + sum(p['cost_usd'] for p in probes):.4f}.",
        "",
        "Methods and limitations: [v34–v38 experiments](../../FULL_MISSION_EXPERIMENTS.md) and [v39–v40 clarity experiments](../../CLARITY_EXPERIMENTS.md).",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "completed": len(rows),
                "validation_strict": sum(r["strict_pass"] for r in validation),
                "validation_completed": len(validation),
                "pending": len(pending),
                "process_failures": failures,
                "audit_problems": {
                    r["tag"]: r["audit_problems"] for r in rows if r["audit_problems"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
