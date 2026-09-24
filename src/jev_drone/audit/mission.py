"""Verify reported objectives against Jev decisions and recorded observations."""

import numpy as np


def audit_reports(result):
    if "course" in result:
        return []
    problems = []
    reports = result["reports"]
    mission = result["world"]["mission"]
    planners = [c for c in result["calls"] if c["role"] == "planner" and not c.get("error")]
    for report in reports:
        matching = [
            c
            for c in planners
            if c["choice"] == report["choice"]
            and c["state"]["stage"] == report["stage"]
            and 0 <= report["time"] - c["response_time"] <= 0.5
        ]
        if not matching:
            problems.append("Objective report without matching Jev planner decision")
        if not report["valid"]:
            continue
        stage = report["stage"]
        if not 0 <= stage < len(mission):
            problems.append("Report has an invalid mission stage")
            continue
        objective = mission[stage]
        observation = report["sensors"]
        pose = np.array(observation["position_estimate"])
        supported = observation["speed"] <= 0.10 and abs(observation["yaw_rate_rps"]) <= 0.10
        if report["choice"] == "inspect":
            visible = [
                d for d in observation["detections"] if d["marker"] == objective.get("marker")
            ]
            target = max(visible, key=lambda d: d["pixels"]) if visible else None
            truth = report["target_ground_truth"]
            supported = (
                supported
                and target is not None
                and observation["front_frame_available"]
                and observation["room"] == objective.get("room")
                and np.linalg.norm(pose - target["position"]) <= 1.5
                and truth is not None
                and np.linalg.norm(np.array(report["position"]) - truth["position"]) <= 1.75
            )
        elif report["choice"] == "dock":
            supported = (
                supported
                and "dock" in objective
                and np.linalg.norm(pose - objective["dock"]) <= 0.45
                and np.linalg.norm(np.array(report["position"]) - result["world"]["start"]) <= 0.55
            )
        else:
            supported = False
        if not supported:
            problems.append("Valid objective report lacks observed or native position evidence")
    valid_stages = [r["stage"] for r in reports if r["valid"]]
    if valid_stages != list(range(result["stage"])):
        problems.append("Mission progress differs from ordered valid reports")
    if result["status"] == "success" and result["stage"] != len(mission):
        problems.append("Success before every mission objective was reported")
    return sorted(set(problems))
