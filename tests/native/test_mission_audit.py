from copy import deepcopy

from jev_drone.audit.mission import audit_reports


def completed_inspection():
    observation = {
        "position_estimate": [9.5, 2.0, 1.5],
        "speed": 0.01,
        "yaw_rate_rps": 0.0,
        "room": "bedroom",
        "front_frame_available": True,
        "detections": [{"marker": "red", "position": [10.7, 2.0, 1.5], "pixels": 50}],
    }
    return {
        "world": {"mission": [{"marker": "red", "room": "bedroom"}], "start": [1.4, 2.0, 1.3]},
        "status": "success",
        "stage": 1,
        "calls": [
            {"role": "planner", "choice": "inspect", "state": {"stage": 0}, "response_time": 9.9}
        ],
        "reports": [
            {
                "stage": 0,
                "choice": "inspect",
                "valid": True,
                "time": 10.0,
                "sensors": observation,
                "position": [9.5, 2.0, 1.5],
                "target_ground_truth": {"position": [10.7, 2.0, 1.5]},
            }
        ],
    }


def test_inspection_requires_model_choice_and_current_front_observation():
    result = completed_inspection()
    assert not audit_reports(result)
    no_choice = deepcopy(result)
    no_choice["calls"] = []
    assert "Objective report without matching Jev planner decision" in audit_reports(no_choice)
    result["reports"][0]["sensors"]["detections"] = []
    assert "Valid objective report lacks observed or native position evidence" in audit_reports(
        result
    )


def test_task_completion_cannot_substitute_for_an_inspection_report():
    result = completed_inspection()
    result["reports"] = []
    assert "Mission progress differs from ordered valid reports" in audit_reports(result)
