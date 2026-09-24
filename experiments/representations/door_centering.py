"""Alignment experiment retained for probes; disabled in the v36 default.

The frozen v37/v38 flight snapshots used these facts. Their full-mission outcomes
did not establish a higher strict pass rate, so the working pilot was restored.
"""

import math
from copy import deepcopy

INSTRUCTIONS = """
During CENTER_IN_OPENING, doorway_centering describes the remaining correction
to the opening's centerline. The far-side waypoint's body right offset can be
near zero because of a small yaw error while the drone is still off-center.
For centering, use the opening correction, not that far endpoint projection.
Positive strafe_to_centerline_m is RIGHT, negative is LEFT. Stop lateral motion
once the centerline is aligned. Finish altitude alignment too, then cross.
Movement violations and braking requirements still apply to every choice.
"""

REFERENCE_INSTRUCTIONS = """
While doorway_centering is present, travel_effect.right describes progress toward
the OPENING CENTERLINE; far_waypoint_right_effect preserves the original effect
toward the distant waypoint. Use the centerline effect during this phase.
"""


def describe(state, criteria, *, local_reference=False):
    if (
        state.get("doorway_phase", {}).get("phase") != "CENTER_IN_OPENING"
        or state.get("active_detour")
        or state.get("at_position")
        or state.get("door_passage_complete")
    ):
        return state, criteria
    alignment = state["doorway_alignment"]
    # Door normals are cardinal. The other horizontal coordinate is lateral.
    required = math.radians(alignment["required_heading_degrees"])
    yaw = math.radians(state["sensors"]["yaw_degrees"])
    lateral_right = -math.cos(yaw) if abs(math.cos(required)) > 0.5 else math.sin(yaw)
    if abs(lateral_right) < 0.9:
        return state, criteria
    remaining = -alignment["centerline_offset_m"] / lateral_right
    ready = state["doorway_phase"]["centerline_aligned"]
    state, criteria = deepcopy(state), dict(criteria)
    state["doorway_centering"] = {
        "strafe_to_centerline_m": round(remaining, 3),
        "centerline_aligned": ready,
        "reference": "Known opening and estimated pose; not the far-side waypoint.",
    }
    for name, effect in state["control_effects"].items():
        speed = effect["resulting_controls"]["right_mps"]
        progress = (
            "stopped"
            if speed == 0
            else "axis_reached_brake"
            if ready
            else "toward_goal"
            if speed * remaining > 0
            else "away_from_goal"
        )
        effect["door_centerline_effect"] = progress
        if speed:
            criteria[name] += " Opening centerline strafe effect: " + progress + "."
        if local_reference:
            effect["far_waypoint_right_effect"] = effect["travel_effect"]["right"]
            effect["travel_effect"]["right"] = progress
    return state, criteria
