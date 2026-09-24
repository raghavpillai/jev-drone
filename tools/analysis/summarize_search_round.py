"""Keep this search round separate from older mixed-version house trials."""

import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from jev_drone.audit.native import audit
from tools.analysis.analyze_house_search import analyze as phases
from tools.analysis.analyze_search_coverage import analyze as coverage

TRIALS = [
    ("baseline-v29", "px4-house-v29-transitions", "house_hidden-5111"),
    ("v30-house", "px4-house-v30-survey", "house_hidden-5111"),
    ("v30-present", "px4-house-v30-diagnostics", "house_search-5111"),
    ("v30-absent", "px4-house-v30-diagnostics", "house_search_absent-5111"),
    ("v31-present", "px4-house-v31-progress", "house_search-5111"),
    ("v32-absent", "px4-house-v32-edges", "house_search_absent-5111"),
    ("v33-absent", "px4-house-v33-position", "house_search_absent-5111"),
    ("v34-absent", "px4-house-v34-turns", "house_search_absent-5111"),
]


def number(value):
    return "—" if value is None else f"{value:.1f}"


def main():
    output = Path("report/px4-search")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, cohort, trial in TRIALS:
        path = Path("results") / cohort / trial / "result.json"
        if not path.exists():
            continue
        folder = output / tag
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "audit.json").exists():
            (folder / "audit.json").write_text(json.dumps(audit(path), indent=2) + "\n")
        with redirect_stdout(io.StringIO()):
            phases(path, folder)
            coverage(path, folder)
        row = json.loads((folder / "audit.json").read_text())
        analysis = json.loads((folder / "analysis.json").read_text())
        visible = json.loads((folder / "coverage.json").read_text())
        search = next((p for p in analysis["phases"] if p["phase"] == "Search"), None)
        selections = []
        last_memory = {}
        with path.with_name("calls.jsonl").open() as stream:
            for line in stream:
                if '"selected_task_kind"' not in line:
                    continue
                call = json.loads(line)
                state = call["state"]
                if state.get("search_memory"):
                    last_memory = state["search_memory"]
                if state["selected_task_kind"] == "search":
                    selections.append(call.get("answers", {}).get("selection", {}).get("choice"))
        counts = Counter(x for x in selections if x)
        row.update(
            tag=tag,
            entered=analysis["entered_target_room"],
            first_seen=analysis["first_target_seen"],
            inspected=analysis["inspected"],
            search_seconds=search["duration"] if search else None,
            search_path_metres=search["path_metres"] if search else None,
            search_choices=dict(counts),
            repeated_search_choices=sum(n - 1 for n in counts.values()),
            search_memory_at_last_plan=last_memory,
            coverage=visible["planes"],
            phases=analysis["phases"],
        )
        row.update(
            immediate_completed_looks=analysis["immediate_completed_looks"],
            longest_immediate_look_streak=analysis["longest_immediate_look_streak"],
        )
        rows.append(row)
    probes = []
    for cohort in (
        "px4-house-v30-search-probe",
        "px4-house-v31-visibility-probe",
        "px4-house-v31-compact-probe",
        "px4-house-v33-position-probe",
    ):
        path = Path("results") / cohort / "calls.jsonl"
        if not path.exists():
            continue
        calls = [json.loads(line) for line in path.read_text().splitlines()]
        probes.append(
            {
                "cohort": cohort,
                "requests": len(calls),
                "cost_usd": sum((c.get("usage") or {}).get("cost", 0) or 0 for c in calls),
                "errors": sum(bool(c.get("error")) for c in calls),
            }
        )
    (output / "results.json").write_text(
        json.dumps({"flights": rows, "probes": probes}, indent=2) + "\n"
    )
    lines = [
        "# Search and path experiment ledger",
        "",
        "All times below are simulated seconds unless stated otherwise. Each row is one development trial, not a reliability estimate. Known-room geometry is permitted; furniture and target truth do not enter Jev requests.",
        "",
        "## Present-target trials",
        "",
        "The room-start trials are easier diagnostics and are not cross-house mission passes. Inspection requires fresh front-camera evidence and independently checked native pose.",
        "",
        "| Trial | Start scope | Room entry | First seen | Inspected | Search duration / path | Final outcome |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if "absent" in r["case"]:
            continue
        scope = "cross-house" if r["case"] == "house_hidden" else "inside workshop"
        lines.append(
            f"| [{r['tag']}]({r['tag']}/phases.png) | {scope} | {number(r['entered'])} | {number(r['first_seen'])} | {number(r['inspected'])} | {number(r['search_seconds'])} s / {number(r['search_path_metres'])} m | {r['status']}, {r['stages']}/{r['mission_stages']} |"
        )
    lines += [
        "",
        "## Absent-target search",
        "",
        "A missing target cannot be successfully inspected. Zero false reports is required, but does not establish complete search. Coverage is a 0.25 m free-space sampling audit of saved FRONT depth frames at 1.6 m height; it is not recognition accuracy or proof of absence. Solid samples are excluded with evaluator geometry that never reaches the model.",
        "",
        "Repeated choices count repeated task labels; looking in the same direction from a different position can be useful. Immediate looks instead count stationary look tasks that finish within 0.5 s with at most 0.05 m translation; a long consecutive streak of the same look exposes a no-op planning loop. Neither metric alone measures search quality.",
        "",
        "| Trial | Sim / wall duration | Saved-depth coverage | Time to 95% / 99% | Repeated search choices | Immediate looks / longest streak | Outcome |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if "absent" not in r["case"]:
            continue
        plane = next(p for p in r["coverage"] if p["height_m"] == 1.6)
        lines.append(
            f"| [{r['tag']}]({r['tag']}/coverage.png) | {r['sim_seconds']:.1f} / {r['wall_seconds']:.1f} | {plane['fraction']:.1%} | {number(plane['seconds_to_95pct'])} / {number(plane['seconds_to_99pct'])} | {r['repeated_search_choices']} | {r['immediate_completed_looks']} / {r['longest_immediate_look_streak']['count']} | {r['status']}, {r['stages']} inspections |"
        )
    lines += [
        "",
        "## Control and provenance audit",
        "",
        "`wall_timeout` preserves the fixed wall watchdog; it is not a completed simulated-time allowance. Accepted Hz includes planning and sensing. Shared software rendering, model-service variability and concurrent probes prevent causal throughput comparisons.",
        "",
        "| Trial | Contacts / rule violations | Audit issues | Accepted control Hz | API p50 / p95 ms | Stale responses | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| [{r['tag']}]({r['tag']}/audit.json) | {r['contacts']} / {r['rule_violations']} | {len(r['audit_problems'])} | {r['accepted_controls_per_wall_second']:.2f} | {r['model_local_p50_ms']:.0f} / {r['model_local_p95_ms']:.0f} | {r['stale_responses']} | ${r['cost_usd']:.4f} |"
        )
    new = [r for r in rows if r["tag"] != "baseline-v29"]
    lines += [
        "",
        f"This round: {len(new)} native trials, {sum(p['requests'] for p in probes)} offline requests, reported cost ${sum(r['cost_usd'] for r in new) + sum(p['cost_usd'] for p in probes):.4f}. The v29 reference is excluded from that cost.",
        "",
        "Protocols, interpretation, implementation boundaries and rejected alternatives: [SEARCH_EXPERIMENTS.md](../../SEARCH_EXPERIMENTS.md).",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "new_trials": len(new),
                "contacts": sum(r["contacts"] for r in new),
                "violations": sum(r["rule_violations"] for r in new),
                "audit_issues": sum(len(r["audit_problems"]) for r in new),
                "offline_requests": sum(p["requests"] for p in probes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
