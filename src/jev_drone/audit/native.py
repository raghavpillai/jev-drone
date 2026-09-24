"""Independent audit of body/yaw assignments, native wire format and front vision."""

import hashlib
import json
import math

import numpy as np

from jev_drone.audit.mission import audit_reports
from jev_drone.audit.routes import audit_routes


def audit(path):
    result = json.loads(path.read_text())
    config = result["world"]["config"]
    lease = config.get("command_lease_seconds", 0.6)

    def clearance_threshold(requested, measured, observation):
        base = (
            config.get("full_clearance_m", 1.3)
            if abs(requested) > 0.25
            else config.get("creep_clearance_m", config.get("slow_clearance_m", 0.4))
            if abs(requested) <= 0.1
            else config.get("slow_clearance_m", 0.4)
        )
        if config.get("stopping_margin"):
            speed = max(abs(requested), abs(measured))
            age = observation.get("frame_age_seconds")
            base = max(
                base,
                speed
                * (
                    (age if age is not None else 0.65)
                    + 2 * lease
                    + config.get("braking_settling_allowance_s", 0.0)
                )
                + speed**2
                + 0.08,
            )
        return base

    problems, violations = audit_reports(result), []
    for name, digest in json.loads((path.parent / "manifest.json").read_text())["sha256"].items():
        if hashlib.sha256((path.parent / "source" / name).read_bytes()).hexdigest() != digest:
            problems.append("Changed source snapshot: " + name)
    calls = result["calls"]
    journal = [json.loads(line) for line in (path.parent / "calls.jsonl").read_text().splitlines()]
    mutable_snapshots = 0
    if len(journal) != len(calls):
        problems.append("Request journal length differs from result")
    for call, recorded in zip(calls, journal):
        if call.get("answers") != recorded.get("answers") or call.get("model") != recorded.get(
            "model"
        ):
            problems.append("Result differs from raw Jev response journal")
        mutable_snapshots += call["state"] != recorded["state"]
        # Earlier result exports retained references to evolving search memory.
        # The append-only request journal always preserves what Jev actually saw.
        call["state"] = recorded["state"]
    problems.extend(audit_routes(calls, result["tasks"]))
    controls = [c for c in calls if c["role"] == "control"]
    accepted = [c for c in controls if c.get("accepted")]
    modern = config.get("control_schema") == "joystick-v1"
    channels = (
        ("forward_mps", "right_mps", "up_mps", "yaw_rate_rps")
        if modern
        else ("forward_mps", "up_mps", "yaw_rate_rps")
    )
    for index, call in enumerate(calls):
        if not call.get("error"):
            if not (call.get("model") or "").startswith("typesafe/jev-"):
                problems.append("Non-Jev or missing model provenance")
            if call["choice"] != call["answers"]["selection"]["choice"]:
                problems.append("Returned Jev choice was changed")
        kind = call["state"].get("selected_task_kind")
        if call["role"] == "planner" and kind is not None:
            previous = calls[index - 1] if index else {}
            if (
                previous.get("role") != "planner_mode"
                or previous.get("choice") != kind
                or previous.get("error")
                or previous.get("state", {}).get("stage") != call["state"].get("stage")
            ):
                problems.append(
                    "Concrete planner task lacks a matching preceding Jev task-kind decision"
                )
        observation = call["state"]["sensors"]
        if set(observation["semantic_camera_names"]) - {"east"}:
            problems.append("Auxiliary camera exposed semantic observations")
        if observation["detections"] and not observation["front_frame_available"]:
            problems.append("Detections without a front frame")
    for call in accepted:
        if call["latency_seconds"] > lease:
            problems.append("Late control accepted")
        name = call["choice"]
        # Reconstruct the expected assignment from its public action name.
        patch = {}
        if name == "brake" or name.startswith(("detour_", "brake_and_", "straight_")):
            patch = dict.fromkeys(channels, 0.0)
        if name in ("stop_forward", "stop_strafe", "stop_vertical", "stop_pan"):
            patch = {
                dict(
                    stop_forward="forward_mps",
                    stop_strafe="right_mps",
                    stop_vertical="up_mps",
                    stop_pan="yaw_rate_rps",
                )[name]: 0.0
            }
        elif name in ("up", "down", "straight_up", "straight_down", "up_creep", "down_creep"):
            patch["up_mps"] = (0.1 if name.endswith("_creep") else 0.25) * (
                1 if "up" in name else -1
            )
        elif name.startswith("strafe_") or name in ("straight_left_slow", "straight_right_slow"):
            patch["right_mps"] = (
                0.1
                if name.endswith("_creep")
                else 0.25
                if name.endswith("_slow")
                else config["speed"]
            ) * (-1 if "left" in name else 1)
        elif "pan_left" in name or "pan_right" in name:
            patch["yaw_rate_rps"] = (
                0.08 if name.endswith("_creep") else 0.2 if name.endswith("_slow") else 0.6
            ) * (1 if "left" in name else -1)
        elif name.endswith(
            (
                "forward_slow",
                "forward_full",
                "backward_slow",
                "backward_full",
                "forward_creep",
                "backward_creep",
            )
        ):
            patch["forward_mps"] = (
                0.1
                if name.endswith("_creep")
                else 0.25
                if name.endswith("_slow")
                else config["speed"]
            ) * (-1 if "backward" in name else 1)
        if config.get("pilot_representation", "").startswith("intent-v"):
            if name != "keep_controls" and not name.startswith("stop_"):
                patch = {**dict.fromkeys(channels, 0.0), **patch}
            if name.startswith("arc_"):
                parts = name.split("_")
                pan = parts[-1]
                if parts[1] == "strafe":
                    patch["right_mps"] = 0.25 if parts[2] == "right" else -0.25
                else:
                    patch["forward_mps"] = 0.25 if parts[1] == "forward" else -0.25
                patch["yaw_rate_rps"] = 0.2 if pan == "left" else -0.2
        if config.get("patch_encoding") == "changed_channels":
            current = call["state"]["current_controls"]["body_controls"]
            patch = {key: value for key, value in patch.items() if current[key] != value}
        effect = call["state"]["control_effects"][name]
        receipt = call["control_receipt"]
        expected_values = {**call["state"]["current_controls"]["body_controls"], **patch}
        if (
            effect["patch"] != patch
            or receipt["patch"] != patch
            or effect["resulting_controls"] != expected_values
            or receipt["before"] != call["state"]["current_controls"]
            or receipt["after"]["body_controls"] != expected_values
            or receipt["expected_revision"] != receipt["before"]["revision"]
            or not receipt["applied"]
        ):
            problems.append("Applied patch differs from Jev selection or supplied control revision")
        observation = call["state"]["sensors"]
        reasons = []
        gap = observation.get("nearest_observed_body_gap_m")
        if expected_values["yaw_rate_rps"] and gap is not None and gap < 0.05:
            reasons.append("rotation_close_to_observed_obstacle")
        for channel, positive, negative in (
            ("forward_mps", "forward", "backward"),
            ("right_mps", "right", "left"),
            ("up_mps", "up", "down"),
        ):
            value = expected_values.get(channel, 0.0)
            if value:
                direction = positive if value > 0 else negative
                depth = observation["body_clearance_m"][direction]
                if depth is None or depth < clearance_threshold(
                    value, observation["body_velocity_mps"][positive], observation
                ):
                    reasons.append("insufficient_or_unknown_" + direction + "_depth")
                measured = observation["body_velocity_mps"][positive]
                if abs(measured) > 0.15 and value * measured < 0:
                    reasons.append("must_brake_before_reversing_" + positive)
        if math.hypot(
            expected_values["forward_mps"], expected_values.get("right_mps", 0.0)
        ) > 0.25 and (
            abs(expected_values["yaw_rate_rps"]) > 0.1 or abs(observation["yaw_rate_rps"]) > 0.12
        ):
            reasons.append("cruise_while_turning")
        if modern:
            translation = [expected_values[c] for c in channels[:3]]
            depth = effect.get("combined_clearance_m")
            measured_speed = np.linalg.norm(list(observation["body_velocity_mps"].values()))
            if sum(v != 0 for v in translation) > 1 and (
                depth is None
                or depth
                < clearance_threshold(np.linalg.norm(translation), measured_speed, observation)
            ):
                reasons.append("insufficient_or_unknown_combined_depth")
        if reasons:
            violations.append({"time": call["time"], "choice": name, "reasons": reasons})
    if violations != result["violations"]:
        problems.append("Movement-rule audit differs from runtime")
    commands = result["commands"]
    events = [
        *commands,
        *[e for e in result["command_expirations"] if e["wall"] >= commands[0]["wall"]],
    ]
    previous = commands[0]["before"]
    revisions = {}
    for event in sorted(events, key=lambda e: (e["wall"], e["before"]["revision"])):
        if event["before"] != previous:
            problems.append("Discontinuous control history")
        values = (
            {**previous["body_controls"], **event["patch"]}
            if "patch" in event
            else dict.fromkeys(channels, 0.0)
        )
        expected = {"body_controls": values, "revision": previous["revision"] + 1}
        if event["after"] != expected:
            problems.append("Unexpected control-state mutation")
        previous = event["after"]
        revisions[previous["revision"]] = event
        if "patch" in event and event["expected_revision"] is not None:
            matching = [
                c
                for c in accepted
                if c["choice"] == event["choice"]
                and c["control_receipt"]["after"] == event["after"]
                and 0 <= event["time"] - c["response_time"] <= 0.12
            ]
            if not matching:
                problems.append("Body update without accepted Jev decision")
        elif "patch" in event and any(values.values()):
            problems.append("Unversioned nonzero movement")
    epoch = result["world"]["mission_start_sim_time"]
    wire = [s for s in result["setpoints"] if s["sim"] >= epoch]
    for sample in wire:
        if sample["position"] is not None or sample["frame"] != 8 or sample["mask"] != 1479:
            problems.append("Unexpected native body control mode")
        values = sample["body_controls"]
        if not np.allclose(
            sample["wire_velocity"],
            [values["forward_mps"], values.get("right_mps", 0.0), -values["up_mps"]],
        ) or not math.isclose(sample["yaw_rate_ned"], -values["yaw_rate_rps"]):
            problems.append("Incorrect forward/up/yaw sign on MAVLink wire")
        expected = [
            values["forward_mps"] * math.cos(sample["yaw_enu"])
            + values.get("right_mps", 0.0) * math.sin(sample["yaw_enu"]),
            values["forward_mps"] * math.sin(sample["yaw_enu"])
            - values.get("right_mps", 0.0) * math.cos(sample["yaw_enu"]),
            values["up_mps"],
        ]
        if not np.allclose(sample["velocity"], expected):
            problems.append("Incorrect heading-relative velocity")
        event = revisions.get(sample["revision"])
        if any(values.values()) and (
            event is None
            or event["after"]["body_controls"] != values
            or sample["wall"] - event["wall"] > lease + 0.01
            or sample["expired"]
        ):
            problems.append("Published body movement lacks a live accepted command")
    latency = [c["latency_seconds"] for c in controls]
    intervals = np.diff([s["wall"] for s in wire])
    heading = np.unwrap([math.atan2(t["camera"][1], t["camera"][0]) for t in result["trace"]])
    values = [c["control_receipt"]["after"]["body_controls"] for c in accepted]
    return {
        "id": result["id"],
        "cohort": path.parent.parent.name,
        "directory": str(path.parent),
        "case": config["case"],
        "status": result["status"],
        "stages": result["stage"],
        "mission_stages": len(result.get("course", result["world"]["mission"])),
        "strict_pass": result["status"] == "success"
        and not result["contacts"]
        and not violations
        and not problems,
        "contacts": len(result["contacts"]),
        "rule_violations": len(violations),
        "audit_problems": sorted(set(problems)),
        "sim_seconds": result["sim_seconds"],
        "wall_seconds": result["wall_seconds"],
        "realtime_factor": result["realtime_factor"],
        "calls": len(calls),
        "accepted_controls": len(accepted),
        "accepted_controls_per_wall_second": len(accepted) / result["wall_seconds"],
        "pan_left_decisions": sum(v["yaw_rate_rps"] > 0 for v in values),
        "pan_right_decisions": sum(v["yaw_rate_rps"] < 0 for v in values),
        "strafe_left_decisions": sum(v.get("right_mps", 0.0) < 0 for v in values),
        "strafe_right_decisions": sum(v.get("right_mps", 0.0) > 0 for v in values),
        "forward_decisions": sum(v["forward_mps"] > 0 for v in values),
        "backward_decisions": sum(v["forward_mps"] < 0 for v in values),
        "simultaneous_pan_translation_decisions": sum(
            v["yaw_rate_rps"] != 0
            and any(
                v.get(c, 0.0) != 0
                for c in (("forward_mps", "right_mps", "up_mps") if modern else ("forward_mps",))
            )
            for v in values
        ),
        "total_physical_yaw_degrees": float(np.sum(np.abs(np.diff(heading))) * 180 / math.pi),
        "model_local_p50_ms": float(np.median(latency) * 1000) if latency else None,
        "model_local_p95_ms": float(np.quantile(latency, 0.95) * 1000) if latency else None,
        "publisher_hz": float(1 / np.mean(intervals)) if len(intervals) else None,
        "publisher_max_gap_seconds": float(max(intervals)) if len(intervals) else None,
        "recovered_telemetry_outages": sum(
            "recovered_wall" in event for event in result.get("telemetry_outages", [])
        ),
        "stale_responses": result["stale_responses"],
        "cost_usd": result["cost_usd"],
        "assigned_channels_per_accepted_control": sum(
            len(c["control_receipt"]["patch"]) for c in accepted
        )
        / len(accepted)
        if accepted
        else None,
        "pilot_representation": config.get("pilot_representation", "initial"),
        "result_states_restored_from_request_journal": mutable_snapshots,
        "models": sorted({c["model"] for c in calls if c.get("model")}),
    }
