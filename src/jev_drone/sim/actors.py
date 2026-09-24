"""Kinematic scene actors, isolated from the flight process and its Python GIL."""

import math
import time

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node


def animate(scene, case, sim_time, mission_epoch, active, running, messages):
    node = Node()
    ready = False
    startup_deadline = time.monotonic() + 8
    try:
        while running.value:
            elapsed = max(0, sim_time.value - mission_epoch.value) if active.value else 0
            positions = {}
            for actor in scene["actors"]:
                positions[f"crossing_actor_{actor['body']}"] = [
                    actor["x"],
                    2 + 1.35 * math.sin(0.65 * elapsed + actor["phase"]),
                    0.9,
                ]
            if case == "moved_target":
                target = scene["targets"]["red"]
                fraction = min(1.0, max(0.0, (elapsed - 18) / 5))
                positions[f"red_marker_{target['body']}"] = [
                    a * (1 - fraction) + b * fraction
                    for a, b in zip(target["position"], [8.2, 3.1, 1.5])
                ]
            request = Pose_V()
            for name, position in positions.items():
                pose = request.pose.add()
                pose.name = name
                pose.position.x, pose.position.y, pose.position.z = position
                pose.orientation.w = 1.0
            ok, response = node.request("/world/jev/set_pose_vector", request, Pose_V, Boolean, 500)
            if not ok or not response.data:
                if not ready and time.monotonic() < startup_deadline:
                    continue
                raise RuntimeError(
                    f"Gazebo moving-entity request failed: transport={ok}, applied={response.data}"
                )
            if not ready:
                messages.put(None)
                ready = True
            time.sleep(0.04)
    except Exception as error:
        messages.put(str(error))
