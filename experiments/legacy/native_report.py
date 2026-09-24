"""Independent audit of model provenance, command causality and trial outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jev_drone.world.world import DIRECTIONS


def audit(path):
    result = json.loads(path.read_text())
    if result["world"]["config"].get("front_controls"):
        from jev_drone.audit.native import audit as audit_front
        return audit_front(path)
    directory = path.parent
    manifest = json.loads((directory/"manifest.json").read_text())
    problems = []
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((directory/"source"/name).read_bytes()).hexdigest() != digest:
            problems.append("Changed source snapshot: "+name)
    calls = result["calls"]
    controls = [c for c in calls if c["role"] == "control"]
    accepted = [c for c in controls if c.get("accepted")]
    for call in calls:
        if call.get("error"):
            continue
        if not (call.get("model") or "").startswith("typesafe/jev-"):
            problems.append("Missing or unexpected model provenance")
        if call.get("choice") != call["answers"]["selection"]["choice"]:
            problems.append("Model choice changed after response")
    violations = []
    for call in accepted:
        if call["latency_seconds"] > .6:
            problems.append("Stale control response accepted")
        choice = call.get("effective_choice", call["choice"])
        if "control_options" in call:
            option = call["control_options"][call["choice"]]
            state = call["state"]
            old = state["current_controls"]["velocity_mps"]
            expected = np.zeros(3)
            if choice.startswith(("slow_", "cruise_")):
                pace, direction = choice.split("_", 1)
                expected = np.array(DIRECTIONS[direction])*(.25 if pace == "slow" else result["world"]["config"]["speed"])
            values = dict(zip(("x", "y", "z"), expected.tolist()))
            patch = {k: v for k, v in values.items() if v != old[k]} if state["control_interface"] == "patch" else values
            if option != {"action": choice, "patch": patch, "resulting_velocity_mps": values}:
                problems.append("Selected control effect differs from independent reconstruction")
            if state["control_effects"][call["choice"]] != {"patch": patch, "resulting_velocity_mps": values}:
                problems.append("Applied effect differs from the effect shown to Jev")
            if state["control_interface"] == "patch":
                receipt = call["control_receipt"]
                if (not receipt["applied"] or receipt["expected_revision"] != state["current_controls"]["revision"]
                        or receipt["before"] != state["current_controls"] or receipt["patch"] != patch):
                    problems.append("Patch applied against an unexpected control state")
        if choice.startswith(("slow_", "cruise_")):
            speed, direction = choice.split("_", 1)
            observation = call["state"]["sensors"]
            depth = observation["clearance_m"][direction]
            velocity = np.array(observation["velocity_estimate"])
            reasons = []
            if depth is None or depth < (.4 if speed == "slow" else 1.3):
                reasons.append("unknown_or_insufficient_observed_clearance")
            if np.linalg.norm(velocity) > .15 and np.dot(velocity/np.linalg.norm(velocity), DIRECTIONS[direction]) < .95:
                reasons.append("changed_axis_while_moving")
            if reasons:
                violations.append({"time": call["time"], "choice": choice, "reasons": reasons})
    if violations != result["violations"]:
        problems.append("Control-rule audit differs from runtime")
    commands = result.get("commands", [])
    # Reconstruct the latch independently from logged assignments and watchdog resets.
    if commands and "after" in commands[0]:
        events = [*commands, *[e for e in result.get("command_expirations", [])
                              if e["wall"] >= commands[0]["wall"]]]
        previous = commands[0]["before"]
        for event in sorted(events, key=lambda e: (e["wall"], e["before"]["revision"])):
            if event["before"] != previous:
                problems.append("Control-state revision chain is discontinuous")
            if "patch" in event:
                values = {**previous["velocity_mps"], **event["patch"]}
                if event["expected_revision"] is not None and event["expected_revision"] != previous["revision"]:
                    problems.append("Stale revision was applied")
                if list(values.values()) != event["velocity"]:
                    problems.append("Published command does not reflect its partial assignment")
            else:
                values = dict.fromkeys(("x", "y", "z"), 0.)
            expected = {"velocity_mps": values, "revision": previous["revision"]+1}
            if event["after"] != expected:
                problems.append("Control state changed beyond selected assignments or expiry reset")
            previous = event["after"]
    for command in commands:
        if command.get("expected_revision") is not None:
            matching_patch = [c for c in accepted if c["choice"] == command["choice"]
                and c.get("control_receipt", {}).get("after") == command["after"]
                and c["control_receipt"]["patch"] == command["patch"]
                and 0 <= command["time"]-c["response_time"] <= .12]
            if not matching_patch:
                problems.append("Partial update without matching accepted Jev receipt")
        if np.linalg.norm(command["velocity"]) == 0:
            continue
        matching = [c for c in accepted if c["choice"] == command["choice"]
                    and 0 <= command["time"]-c["response_time"] <= .12]
        if not matching:
            problems.append("Movement command without corresponding accepted Jev decision")
    epoch = result["world"]["mission_start_sim_time"]
    setpoints = [s for s in result["setpoints"] if s["sim"] >= epoch]
    for setpoint in setpoints:
        if setpoint["position"] is not None:
            problems.append("Position autopilot remained active during Jev mission")
        if np.linalg.norm(setpoint["velocity"]) == 0:
            continue
        preceding = [c for c in commands if c["wall"] <= setpoint["wall"]]
        if not preceding:
            problems.append("Published velocity has no command")
        else:
            command = preceding[-1]
            # Publisher timestamps precede its lock acquisition; command logging
            # follows set_velocity. Permit only the adjacent 5 ms handoff window.
            adjacent = [c for c in commands if 0 < c["wall"]-setpoint["wall"] < .005
                        and np.allclose(c["velocity"], setpoint["velocity"])]
            if not np.allclose(command["velocity"], setpoint["velocity"]) and adjacent:
                command = adjacent[0]
            if not np.allclose(command["velocity"], setpoint["velocity"]):
                problems.append("Published velocity differs from requested command")
            if setpoint["wall"]-command["wall"] > .61:
                problems.append("Published movement outlived its command lease")
    models = sorted({c.get("model") for c in calls if c.get("model")})
    latency = [c["latency_seconds"] for c in controls]
    ages = [c["state"]["sensors"]["frame_age_seconds"]+c["latency_seconds"] for c in accepted
            if c["state"]["sensors"].get("frame_age_seconds") is not None]
    intervals = np.diff([s["wall"] for s in setpoints])
    changes = sum(not np.allclose(c["velocity"], p["velocity"])
                  for p, c in zip(commands, commands[1:]))
    confirmations = sum(c["control_options"][c["choice"]]["resulting_velocity_mps"] == c["state"]["current_controls"]["velocity_mps"]
        for c in accepted if "control_options" in c)
    channel_assignments = sum(len(c["control_options"][c["choice"]]["patch"]) for c in accepted if "control_options" in c)
    return {"id": result["id"], "case": result["world"]["config"]["case"],
            "control_interface": result["world"]["config"].get("control_interface", "full"),
            "status": result["status"], "stages": result["stage"], "mission_stages": len(result["world"]["mission"]),
            "strict_pass": result["status"] == "success" and not result["contacts"] and not violations and not problems,
            "contacts": len(result["contacts"]), "rule_violations": len(violations), "audit_problems": sorted(set(problems)),
            "sim_seconds": result["sim_seconds"], "wall_seconds": result["wall_seconds"],
            "realtime_factor": result["realtime_factor"], "calls": len(calls), "local_calls": len(controls),
            "accepted_controls": len(accepted), "accepted_controls_per_wall_second": len(accepted)/result["wall_seconds"],
            "model_local_p50_ms": float(np.median(latency)*1000) if latency else None,
            "model_local_p95_ms": float(np.quantile(latency, .95)*1000) if latency else None,
            "sensor_to_command_p95_ms": float(np.quantile(ages, .95)*1000) if ages else None,
            "publisher_hz": float(1/np.mean(intervals)) if len(intervals) else None,
            "publisher_max_gap_seconds": float(max(intervals)) if len(intervals) else None,
            "stale_responses": result["stale_responses"], "cost_usd": result["cost_usd"], "models": models,
            "command_vector_changes": changes, "unchanged_control_confirmations": confirmations,
            "published_vector_changes": sum(not np.allclose(a["velocity"], b["velocity"])
                                             for a, b in zip(setpoints, setpoints[1:])),
            "keep_choices": sum(c["choice"] == "keep_controls" for c in accepted),
            "assigned_velocity_channels": channel_assignments,
            "revision_rejections": sum(c.get("rejection_reason") == "control_state_changed_during_request" for c in controls),
            "moving_lease_expirations": sum(any(e.get("before", {}).get("velocity_mps", {}).values())
                                           for e in result.get("command_expirations", [])),
            "mean_local_input_tokens": float(np.mean([c["usage"]["input_tokens"] for c in controls
                                            if "input_tokens" in c.get("usage", {})])) if controls else None,
            "pose_sync_rejections": result["pose_sync_rejections"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit(path) for path in args.paths]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"audit.json").write_text(json.dumps(rows, indent=2))
    table = ["| Trial | Outcome | Stages | Strict pass | Contacts | Rule violations | Seconds | Jev controls/s |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['case']} | {row['status']} | {row['stages']}/{row['mission_stages']} | {'yes' if row['strict_pass'] else 'no'} | {row['contacts']} | {row['rule_violations']} | {row['wall_seconds']:.1f} | {row['accepted_controls_per_wall_second']:.2f} |")
    (args.out/"results.md").write_text("\n".join(table)+"\n")
    print(json.dumps({"trials": len(rows), "completions": sum(r["status"] == "success" for r in rows),
                      "strict_passes": sum(r["strict_pass"] for r in rows), "cost_usd": sum(r["cost_usd"] for r in rows),
                      "audit_problems": {r["id"]: r["audit_problems"] for r in rows if r["audit_problems"]}}, indent=2))


if __name__ == "__main__":
    main()
