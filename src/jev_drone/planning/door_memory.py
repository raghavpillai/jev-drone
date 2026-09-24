"""Remember depth surfaces observed inside the known doorway openings."""

import numpy as np

from jev_drone.world.layout import SMALL

INSTRUCTIONS = """
observed_door_obstructions comes from camera depth, not the room map. A substantial
surface was seen across the central opening of those doors. Prefer another room
connection over repeatedly approaching/crossing an observed obstructed doorway.
For example, a route can pass through an intermediate room before the destination.
Unlisted doors are UNKNOWN, not guaranteed open. A previously obstructed opening
may change; memory includes its observation time. You choose which route to use.
"""


class DoorMemory:
    def __init__(self, layout=SMALL):
        self.layout = layout
        self.obstructions = {}

    def observe(self, cloud, time):
        if not len(cloud):
            return
        for name, center in self.layout.doors.items():
            axis = self.layout.door_axis(name)
            across = 1 - axis
            mask = (
                (np.abs(cloud[:, axis] - center[axis]) <= 0.2)
                & (np.abs(cloud[:, across] - center[across]) < 0.5)
                & (cloud[:, 2] >= 0.8)
                & (cloud[:, 2] < 2.1)
            )
            samples = cloud[mask]
            if not len(samples):
                continue
            columns = np.floor((samples[:, across] - center[across] + 0.5) / 0.2).astype(int)
            rows = np.floor((samples[:, 2] - 0.8) / 0.26).astype(int)
            fraction = len(np.unique(columns + 5 * rows)) / 25.0
            if fraction >= 0.4:
                self.obstructions[name] = {
                    "occupied_opening_fraction": round(fraction, 2),
                    "observed_at": time,
                    "source": "rendered depth surface in central opening",
                }

    def describe(self, state, criteria):
        state, criteria = dict(state), dict(criteria)
        state["observed_door_obstructions"] = {
            name: dict(value) for name, value in self.obstructions.items()
        }
        for door, evidence in self.obstructions.items():
            for name in ("approach_" + door, "cross_" + door):
                if name in criteria:
                    criteria[name] += (
                        " OBSERVED OBSTRUCTION across this door's central opening (depth coverage "
                        + str(evidence["occupied_opening_fraction"])
                        + "). Prefer another room connection instead of retrying this route."
                    )
        return state, criteria
