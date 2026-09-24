"""Explicit two-leg bypass candidates; selecting one never issues flight controls."""

import numpy as np

from jev_drone.perception.depth import observed_collision_distance
from jev_drone.world.layout import SMALL

INSTRUCTIONS = """
A detour choice includes planned_route: first move aside/up/down, then travel
past the obstacle while KEEPING the chosen horizontal offset and the altitude
reached by the first leg. Resume the parent's altitude only after the bypass.
Select the whole route only if it is a useful bypass. The second leg is not
certified clear: inspect depth and choose every movement yourself. While
active_detour is present, its current position is your navigation goal; do not
descend or cut back toward the parent task early. You may need to stop/replan if
the next leg is blocked. This preserves your selected plan, not recorded controls.
route_obstacles marks observed surfaces intersecting each proposed leg. Prefer a
route without an observed blocked leg; a clear first step does not make the next
step useful. No observed blocker means UNKNOWN beyond current observations,
not guaranteed free space. A higher bypass can preserve clearance over furniture.
"""


def candidate(pose, goal, direction, distance, preserve_offset, layout=SMALL):
    offset = np.array(direction) * distance
    first = np.array(pose) + offset
    route = [first.tolist()]
    second = np.array(goal) + offset
    # Pass the obstruction at the altitude reached by the first leg. Offsetting
    # the parent's lower altitude would start descending into the obstacle again.
    second[2] = first[2]
    if preserve_offset and np.linalg.norm(second - first) > 0.4:
        # These are the known building bounds, not hidden furniture geometry.
        lo, hi = layout.bounds
        if np.all(second >= lo + [0.45, 0.45, 0.6]) and np.all(second <= hi - [0.45, 0.45, 0.5]):
            route.append(second.tolist())
    return route


def advance(detour, time):
    """Finish a reached leg of the route Jev selected; no control is generated."""
    remaining = detour.get("remaining_route", [])
    if not remaining:
        return None
    # The next leg starts with no progress history. This object also goes to
    # Jev, so internal infinity sentinels must never enter the JSON request.
    progress = {
        key: value
        for key, value in detour.items()
        if key not in ("best_distance", "best_heading_error")
    }
    return {
        **progress,
        "position": remaining[0],
        "remaining_route": remaining[1:],
        "time": time,
        "leg": detour.get("leg", 0) + 1,
    }


def observed_obstacles(route, pose, frames):
    legs = []
    origin = np.array(pose)
    for target in route:
        vector = np.array(target) - origin
        distance = float(np.linalg.norm(vector))
        hit = (
            observed_collision_distance(frames, origin, vector / distance, 0.45)
            if distance > 0.001
            else float("inf")
        )
        blocked = hit < distance
        legs.append(
            {
                "length_m": round(distance, 2),
                "observed_blocked": blocked,
                "observed_obstacle_at_m": round(hit, 2) if blocked else None,
            }
        )
        origin = np.array(target)
    return legs
