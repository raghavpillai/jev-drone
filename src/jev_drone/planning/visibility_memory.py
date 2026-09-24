"""Front-depth visibility memory over architectural samples, never a route solver."""

import numpy as np

INSTRUCTIONS = """
room_visibility summarizes camera evidence at fixed spatial samples, grouped by
room quadrant and height. unknown samples were never supported by front depth;
they may be empty, occupied, or hidden behind furniture. free samples have been
seen in front of a measured surface; surface samples lie near a measured surface.
This is sparse visibility memory, not an obstacle map, a route, or an object
absence test. Prefer viewpoints that can reveal still-unknown parts of the
requested room, especially a different side of an occluder. A new compass heading
that mostly faces a nearby wall need not be useful just because it is unvisited.
If one side is blocked, change position to see around it; do not repeatedly stare
at the same front face. Choose every move yourself using current clearance.
"""


def sample_depth(samples, depth, origin, forward, right, up, focal):
    relative = samples - origin
    axial = relative @ forward
    safe = np.maximum(axial, 0.001)
    height, width = depth.shape
    u = np.rint(width / 2 + focal * (relative @ right) / safe).astype(int)
    v = np.rint(height / 2 - focal * (relative @ up) / safe).astype(int)
    indices = np.flatnonzero(
        (axial > 0.06) & (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1)
    )
    offsets = np.array([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)])
    patches = depth[v[indices, None] + offsets[:, 0], u[indices, None] + offsets[:, 1]]
    finite = np.isfinite(patches)
    count = finite.sum(axis=1)
    ordered = np.sort(np.where(finite, patches, np.inf), axis=1)
    rank = np.maximum(count - 1, 0) * 0.2
    low = np.floor(rank).astype(int)
    high = np.ceil(rank).astype(int)
    rows = np.arange(len(indices))
    measured = ordered[rows, low].copy()
    interpolate = low != high
    measured[interpolate] += (
        ordered[rows[interpolate], high[interpolate]] - measured[interpolate]
    ) * (rank[interpolate] - low[interpolate])
    free = np.zeros(len(samples), dtype=bool)
    surface = np.zeros(len(samples), dtype=bool)
    free[indices] = (count >= 3) & (measured >= axial[indices] + 0.035)
    surface[indices] = (count >= 3) & (np.abs(measured - axial[indices]) <= 0.12)
    return free, surface


class VisibilityMemory:
    def __init__(self, layout):
        self.layout = layout
        self.rooms = {}
        self.stamps = {}
        for room, (lo, hi) in layout.rooms.items():
            samples = np.array(
                [
                    [x, y, z]
                    for z in (0.6, 1.6, 2.4)
                    for y in np.arange(lo[1] + 0.25, hi[1], 0.5)
                    for x in np.arange(lo[0] + 0.25, hi[0], 0.5)
                ]
            )
            self.rooms[room] = {
                "samples": samples,
                "free": np.zeros(len(samples), dtype=bool),
                "surface": np.zeros(len(samples), dtype=bool),
            }

    def observe(self, frames):
        for frame in frames:
            if frame.camera_name != "east":
                continue
            room = self.layout.room_at(frame.origin)
            if room not in self.rooms or frame.time <= self.stamps.get(room, -1):
                continue
            self.stamps[room] = frame.time
            memory = self.rooms[room]
            free, surface = sample_depth(
                memory["samples"],
                frame.depth,
                frame.origin,
                frame.forward,
                frame.right,
                frame.up,
                frame.focal,
            )
            # Static-house observation memory, not a current collision certificate.
            memory["free"] |= free
            memory["surface"] |= surface
            memory["free"] &= ~memory["surface"]

    def summary(self, room):
        if room not in self.rooms:
            return []
        memory = self.rooms[room]
        samples = memory["samples"]
        lo, hi = map(np.array, self.layout.rooms[room])
        center = (lo + hi) / 2
        rows = []
        for z in (0.6, 1.6, 2.4):
            for north in (False, True):
                for east in (False, True):
                    mask = (
                        (samples[:, 2] == z)
                        & ((samples[:, 0] >= center[0]) == east)
                        & ((samples[:, 1] >= center[1]) == north)
                    )
                    free = int(memory["free"][mask].sum())
                    surface = int(memory["surface"][mask].sum())
                    rows.append(
                        {
                            "region": ("north" if north else "south")
                            + "_"
                            + ("east" if east else "west"),
                            "x_range": [
                                float(center[0] if east else lo[0]),
                                float(hi[0] if east else center[0]),
                            ],
                            "y_range": [
                                float(center[1] if north else lo[1]),
                                float(hi[1] if north else center[1]),
                            ],
                            "height_m": z,
                            "free_samples": free,
                            "surface_samples": surface,
                            "unknown_samples": int(mask.sum()) - free - surface,
                        }
                    )
        return rows
