from copy import deepcopy

from experiments.probes.probe_noop_controls import describe
from jev_drone.control.decision_instructions import braking_facts as baseline_braking_facts


def braking_facts(state, criteria):
    # The release-progress candidate remains an offline experiment, not default.
    return describe(*baseline_braking_facts(state, criteria))


def state():
    zero = dict(forward_mps=0.0, right_mps=0.0, up_mps=0.0, yaw_rate_rps=0.0)
    effects = {
        name: {
            "resulting_controls": dict(zero),
            "movement_violations": [],
            "travel_effect": {"forward": "stopped"},
        }
        for name in (
            "brake",
            "keep_controls",
            "stop_pan",
            "stop_forward",
            "brake_and_replan",
            "detour_up",
        )
    }
    effects["forward_creep"] = {
        "resulting_controls": {**zero, "forward_mps": 0.1},
        "movement_violations": [],
        "travel_effect": {"forward": "toward_goal"},
    }
    return {
        "current_controls": {"body_controls": zero},
        "at_position": False,
        "sensors": {"speed": 0.0, "yaw_rate_rps": 0.0},
        "control_effects": effects,
    }


def test_releasing_an_already_stopped_axis_is_explicit_without_removing_choices():
    original = state()
    before = deepcopy(original)
    criteria = dict.fromkeys(original["control_effects"], "Choice.")
    updated, described = braking_facts(original, criteria)
    assert original == before
    assert updated["neutral_and_settled_away_from_goal"]
    assert set(described) == set(criteria)
    for name in ("brake", "keep_controls", "stop_pan", "stop_forward"):
        assert (
            updated["control_effects"][name]["position_progress"]
            == "already_stopped_no_position_progress"
        )
    for name in ("brake_and_replan", "detour_up", "forward_creep"):
        assert "position_progress" not in updated["control_effects"][name]


def test_braking_arrival_and_ready_reports_do_not_get_a_progress_penalty():
    for override in (
        {"sensors": {"speed": 0.26, "yaw_rate_rps": 0.0}},
        {"sensors": {"speed": 0.0, "yaw_rate_rps": 0.2}},
        {"at_position": True},
        {"mission_progress": {"target_in_inspection_range": True}},
        {"mission_progress": {"dock_in_range": True}},
    ):
        original = {**state(), **override}
        updated, _ = braking_facts(original, dict.fromkeys(original["control_effects"], "Choice."))
        assert not updated["neutral_and_settled_away_from_goal"]
        assert "position_progress" not in updated["control_effects"]["keep_controls"]


def test_no_progress_is_requested_when_the_only_goalward_motion_is_unsafe():
    original = state()
    original["control_effects"]["forward_creep"]["movement_violations"] = [
        "insufficient_or_unknown_forward_depth"
    ]
    updated, criteria = braking_facts(
        original, dict.fromkeys(original["control_effects"], "Choice.")
    )
    assert not updated["neutral_and_settled_away_from_goal"]
    assert criteria["forward_creep"].startswith("DO NOT SELECT")


def test_empty_patch_that_keeps_motion_is_not_mistaken_for_a_stopped_release():
    original = state()
    original["current_controls"]["body_controls"]["forward_mps"] = 0.25
    original["control_effects"]["keep_controls"].update(
        patch={}, resulting_controls=dict(original["current_controls"]["body_controls"])
    )
    updated, _ = braking_facts(original, dict.fromkeys(original["control_effects"], "Choice."))
    assert not updated["neutral_and_settled_away_from_goal"]
    assert "position_progress" not in updated["control_effects"]["keep_controls"]
    assert updated["control_effects"]["keep_controls"]["resulting_controls"]["forward_mps"] == 0.25
