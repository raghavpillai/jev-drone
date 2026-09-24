from copy import deepcopy

from jev_drone.control.goal_phase import describe


def state():
    return {
        "active_detour": {"position": [2.0, 2.0, 1.3]},
        "required_heading_degrees": None,
        "requested_heading_degrees": 90.0,
        "goal_bearing_error_degrees": 40.0,
        "horizontal_distance": 0.15,
        "goal_tolerance_m": {"horizontal": 0.25, "vertical": 0.15},
        "body_goal_offset_m": {"up": -1.49},
        "at_position": False,
        "control_effects": {
            "down": {
                "turn_effect": "heading_stays_unaligned",
                "movement_violations": [],
                "resulting_controls": {"yaw_rate_rps": 0.0, "up_mps": -0.25},
            },
            "pan_left": {
                "turn_effect": "turn_toward_heading",
                "movement_violations": [],
                "resulting_controls": {"yaw_rate_rps": 0.6, "up_mps": 0.0},
            },
        },
    }


def test_unfinished_vertical_bypass_does_not_request_parent_inspection_heading():
    original = state()
    before = deepcopy(original)
    criteria = {n: e["turn_effect"] for n, e in original["control_effects"].items()}
    updated, choices = describe(original, criteria)
    assert original == before
    assert updated["bypass_position_components"]["remaining_altitude_m"] == -1.49
    assert not updated["bypass_position_components"]["altitude_reached"]
    assert updated["requested_heading_degrees"] is None
    assert not updated["at_position"]
    assert choices["down"] == "hold_aligned_heading"
    assert choices["pan_left"] == "turn_away_from_alignment"
    for name in criteria:
        assert (
            updated["control_effects"][name]["resulting_controls"]
            == original["control_effects"][name]["resulting_controls"]
        )


def test_door_heading_and_parent_task_survive_outside_the_bypass_phase():
    for override in (
        {"required_heading_degrees": 0.0},
        {"active_detour": None, "body_goal_offset_m": {"up": 0.0}},
        {"horizontal_distance": 0.8},
    ):
        original = {**state(), **override}
        updated, choices = describe(original, {"down": "heading_stays_unaligned"})
        assert updated == original
        assert choices == {"down": "heading_stays_unaligned"}


def test_plain_waypoint_also_defers_final_look_until_altitude_is_reached():
    original = {**state(), "active_detour": None}
    before = deepcopy(original)
    criteria = {n: e["turn_effect"] for n, e in original["control_effects"].items()}
    updated, choices = describe(original, criteria)
    assert original == before
    assert updated["waypoint_position_components"]["remaining_altitude_m"] == -1.49
    assert updated["requested_heading_degrees"] is None
    assert choices["down"] == "hold_aligned_heading"
    assert choices["pan_left"] == "turn_away_from_alignment"
    assert choices.keys() == criteria.keys()


def test_altitude_phase_never_erases_a_depth_violation():
    original = state()
    original["control_effects"]["down"]["movement_violations"] = [
        "insufficient_or_unknown_down_depth"
    ]
    criteria = {
        n: "DO NOT SELECT: " + e["turn_effect"] for n, e in original["control_effects"].items()
    }
    updated, choices = describe(original, criteria)
    assert updated["control_effects"]["down"]["movement_violations"] == [
        "insufficient_or_unknown_down_depth"
    ]
    assert choices["down"].startswith("DO NOT SELECT:")
