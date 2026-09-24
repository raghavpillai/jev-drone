"""Expose permitted short axis corrections; does not change motion constraints."""

import math
from copy import deepcopy

INSTRUCTIONS = """
Read near_goal_axis_progress when present. goal_path_status concerns the exact
3D diagonal toward a waypoint; it does not override the clearance of a different
body-axis motion. Close to the goal, a tiny already-satisfied vertical/lateral
offset can make that diagonal unhelpful. The listed controls have no movement
violations and move an unfinished position component toward its goal without
moving another component away. Consider one of these short corrections before
starting a large detour. Reassess clearance and remaining distance after moving;
the list does not certify the entire route or guarantee final arrival. Stop an
axis once reached, and obey braking/heading constraints. Every choice is yours.
"""


def describe(state, criteria):
    offset = state["body_goal_offset_m"]
    if (
        state["at_position"]
        or state.get("goal_path_status") not in ("BLOCKED", "UNKNOWN")
        or state.get("goal_endpoint_observed_occupied")
        or state.get("required_heading_degrees") is not None
        or math.sqrt(sum(value * value for value in offset.values())) > 1.0
    ):
        return state, criteria
    limits = state["goal_tolerance_m"]
    horizontal_unfinished = state["horizontal_distance"] > limits["horizontal"]
    unfinished = {
        "forward": horizontal_unfinished,
        "right": horizontal_unfinished,
        "up": abs(offset["up"]) > limits["vertical"],
    }
    candidates = []
    for name, effect in state["control_effects"].items():
        if effect["movement_violations"] or name.startswith("detour_"):
            continue
        controls = effect["resulting_controls"]
        if controls["yaw_rate_rps"]:
            continue
        moving = [
            axis
            for axis, channel in [
                ("forward", "forward_mps"),
                ("right", "right_mps"),
                ("up", "up_mps"),
            ]
            if controls[channel]
        ]
        if (
            len(moving) == 1
            and unfinished[moving[0]]
            and effect["travel_effect"][moving[0]] == "toward_goal"
        ):
            candidates.append(name)
    if not candidates:
        return state, criteria
    state, criteria = deepcopy(state), dict(criteria)
    state["near_goal_axis_progress"] = {
        "permitted_controls": candidates,
        "scope": "Short body-axis corrections supported by current depth; not proof the complete diagonal is clear.",
    }
    for name in candidates:
        criteria[name] += (
            " Permitted short axis progress toward the unfinished position goal; consider before a large bypass."
        )
    return state, criteria
