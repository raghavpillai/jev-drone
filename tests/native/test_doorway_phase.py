from copy import deepcopy

from jev_drone.control.doorway_phase import describe


def test_diagonal_blockage_does_not_skip_unfinished_altitude_alignment():
    state = {
        "doorway_alignment": {
            "heading_error_degrees": 0.0,
            "centerline_offset_m": 0.03,
            "altitude_error_m": 0.8,
        },
        "goal_tolerance_m": {"vertical": 0.15},
        "goal_path_status": "BLOCKED",
        "control_effects": {
            "down": {"movement_violations": []},
            "forward_slow": {"movement_violations": ["blocked"]},
        },
    }
    before = deepcopy(state)
    criteria = {
        "down": "descend",
        "forward_slow": "move",
        "brake": "stop",
        "brake_and_replan": "replan",
    }
    enriched, choices = describe(state, criteria)
    assert enriched["doorway_phase"]["phase"] == "CENTER_IN_OPENING"
    assert enriched["doorway_phase"]["diagonal_goal_path_is_not_the_aligned_crossing"]
    assert enriched["control_effects"] == state["control_effects"]
    assert state == before and choices == criteria


def test_turn_then_center_then_cross_follows_measured_errors():
    state = {
        "doorway_alignment": {
            "heading_error_degrees": 90.0,
            "centerline_offset_m": 0.4,
            "altitude_error_m": 0.0,
        },
        "goal_tolerance_m": {"vertical": 0.15},
    }
    assert describe(state, {})[0]["doorway_phase"]["phase"] == "FACE_OPENING"
    state["doorway_alignment"]["heading_error_degrees"] = 0.0
    assert describe(state, {})[0]["doorway_phase"]["phase"] == "CENTER_IN_OPENING"
    state["doorway_alignment"]["centerline_offset_m"] = 0.1
    assert describe(state, {})[0]["doorway_phase"]["phase"] == "CROSS_OPENING"


def test_phase_does_not_replace_a_selected_bypass_or_non_door_task():
    assert describe({"task": {"name": "up"}}, {}) == ({"task": {"name": "up"}}, {})
    state = {
        "doorway_alignment": {"heading_error_degrees": 90.0},
        "active_detour": {"position": [1, 2, 3]},
    }
    assert describe(state, {}) == (state, {})
