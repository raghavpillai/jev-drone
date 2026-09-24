"""Sparse measured-space memory. No access to simulator obstacles or hidden objects."""

import math

from experiments.legacy.representation_experiment import DIRECTIONS, label
from experiments.legacy.sim import norm, subtract, unit


class ObservedMap:
    def __init__(self, resolution=0.5):
        self.resolution = resolution
        self.cells = {}
        self.landmarks = []

    def cell(self, position):
        return tuple(math.floor(p / self.resolution) for p in position)

    def record(self, position, readings, now, cap=3.0):
        def add(point, kind):
            key = self.cell(point)
            item = self.cells.setdefault(key, {"free": 0, "occupied": 0, "last_seen": now})
            item[kind] += 1
            item["last_seen"] = now

        for direction in DIRECTIONS:
            value = readings.get(label(direction))
            if value is None:
                continue
            vector = unit(direction)
            for step in range(1, max(1, int((value - 0.25) / self.resolution) + 1)):
                add(tuple(p + v * step * self.resolution for p, v in zip(position, vector)), "free")
            if value < cap - 0.01:
                add(tuple(p + v * value for p, v in zip(position, vector)), "occupied")
        if not self.landmarks or norm(subtract(position, self.landmarks[-1]["xyz"])) >= 1.0:
            self.landmarks.append(
                {
                    "id": f"visited_{len(self.landmarks)}",
                    "xyz": [round(p, 2) for p in position],
                    "seen_at": round(now, 2),
                }
            )

    def evidence(self, position):
        item = self.cells.get(self.cell(position))
        if not item:
            return "unknown"
        if item["free"] and item["occupied"]:
            return "conflicting"
        return "occupied_evidence" if item["occupied"] else "free_evidence"

    def describe(self, position, now):
        nearby = []
        for key, item in self.cells.items():
            point = [(k + 0.5) * self.resolution for k in key]
            distance = norm(subtract(point, position))
            if distance <= 4.0:
                nearby.append(
                    (
                        distance,
                        {
                            "relative_xyz": [round(p - q, 2) for p, q in zip(point, position)],
                            "evidence": self.evidence(point),
                            "age_s": round(now - item["last_seen"], 2),
                        },
                    )
                )
        nearby.sort(key=lambda pair: pair[0])
        return {
            "meaning": "Sparse ray samples, not certified free cells. Unlisted space is UNKNOWN. Evidence can be stale or conflicting; no route computed.",
            "cell_width_m": self.resolution,
            "recorded_cells": len(self.cells),
            "nearby_samples": [value for _, value in nearby[:48]],
            "visited_positions": self.landmarks[-24:],
        }


def recovery_options(position, memory):
    points = {"resume": None}
    criteria = {
        "resume": "Keep pursuing the mission waypoint; the local controller can still make progress."
    }
    for direction in DIRECTIONS:
        if direction[2] or sum(abs(v) for v in direction) != 1:
            continue
        name = "explore_" + label(direction)
        point = [round(p + 2.5 * v, 2) for p, v in zip(position, direction)]
        points[name] = point
        criteria[name] = (
            f"Temporary exploration waypoint {point}; map evidence at endpoint: {memory.evidence(point)}. This does not guarantee a clear path."
        )
    for item in memory.landmarks[-12:]:
        if norm(subtract(item["xyz"], position)) > 1.0:
            points[item["id"]] = item["xyz"]
            criteria[item["id"]] = (
                f"Return toward previously visited position {item['xyz']}, observed at {item['seen_at']} s. The straight path may be obstructed."
            )
    return points, criteria
