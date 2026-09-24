"""Expose a blocked altitude subgoal without selecting or suppressing controls."""

INSTRUCTIONS = """
If altitude_access says requested_vertical_motion_permitted=false, the remaining
height CANNOT be reached by moving vertically from this observation. The fact that
horizontal_reached=true does not make up/down safe. Stop vertical motion. Choose
an observed viable lateral bypass or brake_and_replan so another approach can be
chosen. Never keep correcting height into a surface just because it is closer to
the goal height. A goal coordinate is not permission to enter an obstacle.
"""


def describe(state, criteria):
    remaining = state["body_goal_offset_m"]["up"]
    if abs(remaining) <= state["goal_tolerance_m"]["vertical"] or state["at_position"]:
        return state, criteria
    names = ["up", "up_creep"] if remaining > 0 else ["down", "down_creep"]
    effects = state["control_effects"]
    permitted = [
        name for name in names if name in effects and not effects[name]["movement_violations"]
    ]
    state = dict(state)
    state["altitude_access"] = {
        "remaining_m": remaining,
        "direction": "up" if remaining > 0 else "down",
        "horizontal_reached": state["horizontal_distance"]
        <= state["goal_tolerance_m"]["horizontal"],
        "requested_vertical_motion_permitted": bool(permitted),
        "permitted_vertical_controls": permitted,
        "constraints": {
            name: effects[name]["movement_violations"] for name in names if name in effects
        },
    }
    if not permitted:
        criteria = dict(criteria)
        criteria["brake_and_replan"] += (
            " The requested height correction is blocked by observed clearance; stopping and asking for another approach is appropriate."
        )
    return state, criteria
