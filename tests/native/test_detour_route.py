import json

import numpy as np

from jev_drone.planning.detour_route import advance, candidate, observed_obstacles


def test_vertical_bypass_keeps_altitude_until_past_the_obstacle():
    route = candidate([3.0, 2.0, 1.3], [5.0, 2.0, 1.3], [0.0, 0.0, 1.0], 1.3, True)
    assert np.allclose(route, [[3.0, 2.0, 2.6], [5.0, 2.0, 2.6]])
    selected = {
        "position": route[0],
        "remaining_route": route[1:],
        "selected_by": "jev",
        "time": 1.0,
    }
    second = advance(selected, 4.0)
    assert second["position"] == route[1] and second["selected_by"] == "jev"
    assert selected["position"] == route[0]
    assert advance(second, 8.0) is None


def test_lateral_bypass_keeps_offset_and_does_not_choose_outside_building():
    assert candidate([3.0, 3.0, 1.3], [5.0, 3.0, 1.3], [0.0, 1.0, 0.0], 1.3, True) == [
        [3.0, 4.3, 1.3],
        [5.0, 4.3, 1.3],
    ]
    assert candidate([3.0, 6.0, 1.3], [5.0, 7.0, 1.3], [0.0, 1.0, 0.0], 1.3, True) == [
        [3.0, 7.3, 1.3]
    ]


def test_advancing_leg_resets_progress_without_nonfinite_request_values():
    selected = {
        "position": [2.0, 2.0, 2.6],
        "remaining_route": [[5.0, 2.0, 2.6]],
        "best_distance": 0.1,
        "best_heading_error": 0.02,
        "selected_by": "jev",
        "leg": 0,
    }
    second = advance(selected, 4.0)
    assert "best_distance" not in second and "best_heading_error" not in second
    assert selected["best_distance"] == 0.1
    assert json.loads(json.dumps(second, allow_nan=False))["leg"] == 1


def test_forward_back_reposition_does_not_install_an_unasked_parallel_route():
    assert candidate([3.0, 3.0, 1.3], [5.0, 3.0, 1.3], [-1.0, 0.0, 0.0], 0.6, False) == [
        [2.4, 3.0, 1.3]
    ]


def test_clear_first_step_can_have_an_observed_blocked_second_leg():
    from types import SimpleNamespace

    # These are measured points, not a hidden scene/object model.
    frames = [SimpleNamespace(clearance_cloud=np.array([[3.6, 3.0, 1.3]]))]
    side = observed_obstacles([[2.0, 3.3, 1.3], [5.0, 3.3, 1.3]], [2.0, 2.0, 1.3], frames)
    above = observed_obstacles([[2.0, 2.0, 2.6], [5.0, 2.0, 2.6]], [2.0, 2.0, 1.3], frames)
    assert not side[0]["observed_blocked"] and side[1]["observed_blocked"]
    assert not any(leg["observed_blocked"] for leg in above)
    assert (
        observed_obstacles([[3.0, 2.0, 1.3]], [2.0, 2.0, 1.3], [])[0]["observed_obstacle_at_m"]
        is None
    )


def test_upward_bypass_does_not_descend_toward_a_lower_parent_goal_mid_crossing():
    route = candidate([9.0, 3.0, 2.3], [7.0, 2.0, 1.3], [0.0, 0.0, 1.0], 0.9, True)
    assert np.allclose(route, [[9.0, 3.0, 3.2], [7.0, 2.0, 3.2]])


def test_lateral_bypass_preserves_flight_altitude_until_past_the_obstruction():
    route = candidate([9.0, 3.0, 2.9], [7.0, 2.0, 1.3], [0.0, 1.0, 0.0], 0.6, True)
    assert np.allclose(route, [[9.0, 3.6, 2.9], [7.0, 2.6, 2.9]])


def test_medium_upward_route_fits_between_observed_partition_and_ceiling():
    from types import SimpleNamespace

    # A 2.7 m partition requires center height above 3.15 m for the 0.45 m body.
    frames = [SimpleNamespace(clearance_cloud=np.array([[8.0, 2.0, 2.7], [9.0, 2.0, 4.0]]))]
    pose, goal, direction = [9.0, 2.0, 2.3], [7.0, 2.0, 1.3], [0.0, 0.0, 1.0]
    short = candidate(pose, goal, direction, 0.6, True)
    medium = candidate(pose, goal, direction, 0.9, True)
    full = candidate(pose, goal, direction, 1.3, True)
    assert observed_obstacles(short, pose, frames)[1]["observed_blocked"]
    assert not any(leg["observed_blocked"] for leg in observed_obstacles(medium, pose, frames))
    assert observed_obstacles(full, pose, frames)[0]["observed_blocked"]
