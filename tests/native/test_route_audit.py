from copy import deepcopy

from jev_drone.audit.routes import audit_routes


def test_active_route_legs_require_the_actual_accepted_model_selection():
    route = [[3.0, 2.0, 2.6], [5.0, 2.0, 2.6]]
    selected = {
        "role": "control",
        "time": 1.0,
        "choice": "detour_up",
        "accepted": True,
        "state": {"control_effects": {"detour_up": {"planned_route": route}}},
    }
    later = {
        "role": "control",
        "accepted": False,
        "state": {
            "active_detour": {
                "planned_route": route,
                "position": route[1],
                "remaining_route": [],
                "leg": 1,
                "decision_time": 1.0,
                "choice": "detour_up",
            }
        },
    }
    assert not audit_routes([selected, later])
    assert audit_routes([later])
    wrong = deepcopy(later)
    wrong["state"]["active_detour"]["position"][0] = 8.0
    assert audit_routes([selected, wrong])
    assert audit_routes([{**selected, "accepted": False}, later])


def test_replanning_outcome_requires_an_accepted_model_handoff():
    task = {
        "name": "cross_living_bedroom",
        "time": 1.0,
        "ended": 5.0,
        "outcome": "replan_requested",
    }
    call = {
        "role": "control",
        "time": 4.5,
        "response_time": 5.0,
        "accepted": True,
        "choice": "brake_and_replan",
        "state": {"task": {"name": task["name"]}, "control_effects": {"brake_and_replan": {}}},
    }
    assert not audit_routes([call], [task])
    assert audit_routes([{**call, "accepted": False}], [task])
