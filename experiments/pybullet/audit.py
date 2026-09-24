"""Independent reconstruction of reports, model identity and action contracts."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from jev_drone.world.world import DIRECTIONS, ROOMS


def audit_episode(result):
    issues = []
    violations = []
    motion_risk_ignored = []
    for call in result["calls"]:
        if call.get("model") and "jev" not in call["model"].lower():
            issues.append("non_jev_model")
        if call.get("choice") and call["answers"]["selection"]["choice"] != call["choice"]:
            issues.append("choice_changed_after_model")
        if call["role"] != "control" or not call.get("accepted"):
            continue
        if call["latency_seconds"] > .6:
            issues.append("stale_response_accepted")
        choice = call["choice"]
        if not choice.startswith(("cruise_", "slow_")):
            continue
        speed, direction = choice.split("_", 1)
        sensor = call["state"]["sensors"]
        clearance = sensor["clearance_m"][direction]
        velocity = np.array(sensor["velocity_estimate"])
        reasons = []
        if clearance is None or clearance < (.4 if speed == "slow" else 1.3):
            reasons.append("unknown_or_insufficient_observed_clearance")
        if np.linalg.norm(velocity) > .15 and np.dot(velocity/np.linalg.norm(velocity), DIRECTIONS[direction]) < .95:
            reasons.append("changed_axis_while_moving")
        if reasons:
            violations.append({"time": call["time"], "choice": choice, "reasons": reasons})
        if sensor.get("predicted_risk_by_direction", {}).get(direction) or (sensor.get("current_course_risk") and np.linalg.norm(velocity) > .15):
            motion_risk_ignored.append(call["time"])
    if violations != result["violations"]:
        issues.append("control_audit_disagrees")
    report_checks = []
    mission = result["world"]["mission"]
    geometry = {b["name"]: b for b in result["world"]["geometry"]}
    for report in result["reports"]:
        stage = report["stage"]
        objective = mission[stage]
        sensor = report["sensors"]
        pose = np.array(sensor["position_estimate"])
        actual = np.array(report["position"])
        # Velocity in these experiments is idealized but serialized to 0.01m/s.
        valid = np.linalg.norm(sensor["velocity_estimate"]) <= .159
        if report["choice"] == "dock":
            valid &= "dock" in objective and np.linalg.norm(actual-result["world"]["start"]) <= .55
            valid &= np.linalg.norm(pose-result["world"]["start"]) <= .45
        else:
            marker = objective.get("marker")
            if marker is None:
                valid = False
            else:
                target = np.array(geometry[marker+"_marker"]["center"])
                if result["world"]["config"]["case"] == "moved_target" and marker == "red" and report["time"] >= 18:
                    target = np.array([8.2, 3.1, 1.5])
                valid &= np.linalg.norm(actual-target) <= 1.75
                lo, hi = ROOMS[objective["room"]]
                valid &= lo[0] <= pose[0] <= hi[0] and lo[1] <= pose[1] <= hi[1]
                valid &= any(d["marker"] == marker and np.linalg.norm(pose-d["position"]) <= 1.5 and report["time"]-d["time"] <= .65
                             for d in sensor["detections"])
        report_checks.append(bool(valid))
        if report["valid"] and not valid:
            issues.append("claimed_valid_report_failed_reconstruction")
    if result["status"] == "success" and (len(report_checks) != len(mission) or not all(report_checks) or result["contacts"]):
        issues.append("success_failed_reconstruction")
    return {"id": result["id"], "issues": issues, "report_checks": report_checks,
            "audited_violations": len(violations), "moving_risk_ignored_count": len(motion_risk_ignored)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.directories:
        for path in sorted(root.glob("*.json")):
            if path.name in ("manifest.json", "protocol.json"):
                continue
            result = json.loads(path.read_text())
            if "calls" in result:
                rows.append({"path": str(path), **audit_episode(result)})
    output = {"episodes": len(rows), "issues": dict(Counter(i for r in rows for i in r["issues"])), "rows": rows}
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
