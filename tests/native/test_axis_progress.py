from copy import deepcopy

from jev_drone.control.axis_progress import describe


def state():
    effects = {}
    for name, forward, up, violations in [
        ("forward_creep", 0.1, 0.0, []),
        ("down", 0.0, -0.25, []),
        ("backward_slow", -0.25, 0.0, []),
        ("forward_full", 0.5, 0.0, ["insufficient_depth"]),
    ]:
        effects[name] = {
            "movement_violations": violations,
            "resulting_controls": {
                "forward_mps": forward,
                "right_mps": 0.0,
                "up_mps": up,
                "yaw_rate_rps": 0.0,
            },
            "travel_effect": {
                "forward": "toward_goal" if forward > 0 else "away_from_goal",
                "right": "stopped",
                "up": "toward_goal",
            },
        }
    return {
        "at_position": False,
        "goal_path_status": "BLOCKED",
        "horizontal_distance": 0.48,
        "body_goal_offset_m": {"forward": 0.48, "right": 0.0, "up": -0.05},
        "goal_tolerance_m": {"horizontal": 0.35, "vertical": 0.15},
        "control_effects": effects,
    }


def test_diagonal_status_does_not_hide_permitted_axis_progress_or_change_constraints():
    original = state()
    before = deepcopy(original)
    criteria = dict.fromkeys(original["control_effects"], "Original.")
    updated, choices = describe(original, criteria)
    assert original == before and set(choices) == set(criteria)
    assert updated["control_effects"] == original["control_effects"]
    assert updated["goal_path_status"] == "BLOCKED"
    assert updated["near_goal_axis_progress"]["permitted_controls"] == ["forward_creep"]


def test_ready_occupied_fixed_heading_and_far_goals_are_not_promoted():
    for change in [
        {"at_position": True},
        {"goal_path_status": "OBSERVED_CLEAR"},
        {"goal_endpoint_observed_occupied": True},
        {"required_heading_degrees": 180.0},
        {"body_goal_offset_m": {"forward": 1.1, "right": 0.0, "up": 0.0}},
    ]:
        original = {**state(), **change}
        updated, _ = describe(original, dict.fromkeys(original["control_effects"], ""))
        assert updated is original


def test_horizontal_arrival_defers_to_remaining_altitude_and_unsafe_controls_stay_unsafe():
    original = state()
    original["horizontal_distance"] = 0.2
    original["body_goal_offset_m"] = {"forward": 0.2, "right": 0.0, "up": -0.4}
    updated, _ = describe(original, dict.fromkeys(original["control_effects"], ""))
    assert updated["near_goal_axis_progress"]["permitted_controls"] == ["down"]
    original["control_effects"]["down"]["movement_violations"] = ["insufficient_depth"]
    assert describe(original, dict.fromkeys(original["control_effects"], ""))[0] is original
