"""Separate unfinished position altitude from the final camera heading."""
from copy import deepcopy


INSTRUCTIONS = """
Read waypoint_position_components or bypass_position_components. If
horizontal_reached is true and altitude_reached is false, horizontal travel is
finished but the POSITION is not reached. Finish its altitude before facing the
final inspection item or looking into the room. Do not chase a tiny horizontal
offset with repeated turns while an altitude change remains.
Choose allowed up/down movement toward the signed remaining altitude; up/down
also releases pan and horizontal controls. If already stopped, do not repeatedly
brake instead of making this permitted vertical progress. Never violate depth
or braking constraints. The final camera heading is deferred until this position
is reached. These facts do not select a control for you.
"""


def describe(state, criteria):
    if (state.get("required_heading_degrees") is not None
            or state["horizontal_distance"] > state["goal_tolerance_m"]["horizontal"]):
        return state, criteria
    altitude_reached = abs(state["body_goal_offset_m"]["up"]) <= state["goal_tolerance_m"]["vertical"]
    if not state.get('active_detour') and altitude_reached:
        return state, criteria
    state, criteria = deepcopy(state), dict(criteria)
    state["requested_heading_degrees"] = None
    state["goal_bearing_error_degrees"] = 0.
    key = 'bypass_position_components' if state.get('active_detour') else 'waypoint_position_components'
    state[key] = {
        "horizontal_reached": True,
        "altitude_reached": altitude_reached,
        "remaining_altitude_m": state["body_goal_offset_m"]["up"],
        "parent_inspection_heading_deferred": True,
    }
    for name, effect in state["control_effects"].items():
        previous = effect["turn_effect"]
        effect["turn_effect"] = ("turn_away_from_alignment" if effect["resulting_controls"]["yaw_rate_rps"]
                                 else "hold_aligned_heading")
        if name in criteria:
            criteria[name] = criteria[name].replace(previous, effect["turn_effect"])
    return state, criteria
