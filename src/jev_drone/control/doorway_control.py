"""Describe the known doorway centerline; Jev still selects every control."""

from jev_drone.world.layout import SMALL

INSTRUCTIONS = """
For a cross_DOOR task, doorway_alignment describes the known opening. First face
the required crossing heading, then center horizontally and vertically on the
opening using strafe/up/down while HOLDING that heading. Cross straight through
the centerline. Do not pan/arc toward a diagonal shortcut or shift the crossing
line sideways: that aims at the jamb. If aligned and centered but the opening is
observed blocked, stop for replanning; going sideways/up through the wall is not
a bypass. Unknown depth while turning calls for finishing alignment and observing
the opening before moving. Outside a crossing, ordinary detours remain useful.
"""


def heading(task, layout=SMALL):
    door = task.get("name", "").removeprefix("cross_")
    if not task.get("name", "").startswith("cross_") or door not in layout.doors:
        return None
    axis = layout.door_axis(door)
    positive = task["position"][axis] > layout.doors[door][axis]
    return (90.0 if positive else -90.0) if axis else (0.0 if positive else 180.0)


def describe(state, criteria, layout=SMALL):
    task = state["task"]
    required = heading(task, layout)
    if required is None:
        return state, criteria
    state, criteria = dict(state), dict(criteria)
    observation = state["sensors"]
    door = task["name"][6:]
    axis = layout.door_axis(door)
    pose = observation["position_estimate"]
    error = (required - observation["yaw_degrees"] + 180) % 360 - 180
    state["doorway_alignment"] = {
        "heading_error_degrees": round(error, 2),
        "required_heading_degrees": required,
        "centerline_offset_m": round(pose[1 - axis] - layout.doors[door][1 - axis], 3),
        "altitude_error_m": round(pose[2] - task["position"][2], 3),
        "plane_distance_m": round(abs(pose[axis] - layout.doors[door][axis]), 3),
    }
    for name in criteria:
        if name.startswith("detour_"):
            criteria[name] += (
                " During doorway crossing this shifts the route toward a wall/jamb. Prefer aligning with the opening or stopping for replanning."
            )
    return state, criteria
