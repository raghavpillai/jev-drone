"""Decision presentation experiments; every action remains available to Jev."""

from copy import deepcopy

BRAKING = """
SAFETY TAKES PRIORITY OVER PROGRESS. Never select a control whose
movement_violations is nonempty, including keep_controls. A zero requested
velocity does NOT mean the drone has stopped: read measured body_velocity_mps
and yaw_rate_rps. While neutral controls are braking residual movement, keeping
neutral is useful progress, not a hover loop. Wait until the measured velocity
allows a new movement before reversing. Do not start a bypass to escape this
brief braking phase. A forbidden movement toward the goal is still forbidden.
"""


def braking_facts(state, criteria):
    state, criteria = deepcopy(state), dict(criteria)
    current = state["current_controls"]["body_controls"]
    observation = state["sensors"]
    moving = observation["speed"] > 0.10 or abs(observation["yaw_rate_rps"]) > 0.10
    braking = not any(current.values()) and moving
    state["neutral_controls_still_braking"] = braking
    state["holding_position_away_from_goal"] = (
        not any(current.values()) and not moving and not state["at_position"]
    )
    for name, effect in state["control_effects"].items():
        reasons = effect.get("movement_violations", [])
        if reasons:
            criteria[name] = "DO NOT SELECT: violates " + ", ".join(reasons) + ". " + criteria[name]
        if name == "keep_controls" and braking:
            effect["progress"] = "braking_measured_motion"
            criteria[name] = (
                "Keep neutral controls while PX4 brakes measured residual motion. Wait for the drone to stop; this is NOT a stalled hover."
            )
    return state, criteria


TRAVEL = """
When heading_requirement is free_during_translation, facing the position goal is
OPTIONAL. Use forward/back or strafe toward body_goal_offset_m while holding the
current heading. Prefer a straight move over an unnecessary translation/pan arc.
An unaligned nose is not a navigation error: sideways and backward motion work.
Pan when needed to observe the target or search, or when an explicit heading is
required. Complete a final inspection heading after reaching the position.
"""


def heading_facts(state, criteria):
    state, criteria = deepcopy(state), dict(criteria)
    task = state["task"]
    required = state.get("required_heading_degrees") is not None or (
        state["at_position"] and bool(task.get("look") or task.get("look_position"))
    )
    state["heading_requirement"] = "required_now" if required else "free_during_translation"
    if not required:
        for name, effect in state["control_effects"].items():
            old = effect["turn_effect"]
            effect["turn_effect"] = (
                "optional_pan"
                if effect["resulting_controls"]["yaw_rate_rps"]
                else "maintain_heading"
            )
            criteria[name] = criteria[name].replace(old, effect["turn_effect"])
    return state, criteria
