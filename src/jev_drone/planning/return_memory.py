"""Remember observed flight positions; Jev decides whether and how to retrace them."""

import math

import numpy as np

from jev_drone.planning.return_intent import INSTRUCTIONS as INSTRUCTIONS


class ReturnMemory:
    def __init__(self):
        self.points = []
        self.returning = False

    def observe(self, observation, returning):
        if self.returning:
            return
        self.returning = returning
        pose = np.array(observation["position_estimate"])
        previous = self.points[-1] if self.points else None
        heading = observation["yaw_degrees"]
        moved = float(np.linalg.norm(pose - previous["position"])) if previous else math.inf
        turned = abs((heading - previous["yaw_degrees"] + 180) % 360 - 180) if previous else 0.0
        if previous and moved < 0.8 and not (moved >= 0.2 and turned >= 35):
            return
        # Keep a bounded history. Dropped samples never certify a connecting path.
        if len(self.points) >= 256:
            self.points = self.points[::2]
        self.points.append(
            {"position": pose.tolist(), "room": observation["room"], "yaw_degrees": heading}
        )

    def options(self, observation, tasks):
        if not self.returning:
            return {}
        cursor = min(
            (
                t["return_route_index"]
                for t in tasks
                if "return_route_index" in t and t["outcome"] == "arrived"
            ),
            default=len(self.points),
        )
        pose = np.array(observation["position_estimate"])
        candidates = {}
        for index in range(cursor - 1, -1, -1):
            point = self.points[index]
            delta = np.array(point["position"]) - pose
            distance = float(np.linalg.norm(delta))
            if distance > 3.5 or (np.linalg.norm(delta[:2]) <= 0.35 and abs(delta[2]) <= 0.15):
                continue
            candidates[f"retrace_{index}"] = {
                "position": point["position"],
                "room": point["room"],
                "return_route_index": index,
                "purpose": "Previously flown position on return toward launch. Earlier indices lead toward launch; verify present clearance.",
            }
            if len(candidates) == 8:
                break
        return candidates

    def describe(self):
        return {
            "returning": self.returning,
            "recorded_positions": len(self.points),
            "source": "Past estimated drone poses during this mission; no hidden obstacle coordinates or automatic steering.",
        }
