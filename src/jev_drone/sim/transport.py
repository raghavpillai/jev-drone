"""Gazebo subscriptions. Truth channels belong exclusively to the evaluator."""
from collections import deque
from functools import partial
import math
import threading

import numpy as np
from gz.transport13 import Node
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.entity_pb2 import Entity

from jev_drone.sim.scene import CAMERAS


def stamp(message):
    return message.header.stamp.sec+message.header.stamp.nsec*1e-9


class Transport:
    def __init__(self, shared_time=None):
        self.node = Node()
        self.lock = threading.Lock()
        self.time = 0.
        self.shared_time = shared_time
        self.images = {name: {"rgb": deque(maxlen=12), "depth": deque(maxlen=12)} for name in CAMERAS}
        self.truth = None
        self.truth_yaw = 0.
        self.entity_truth = {}
        self.contacts = deque(maxlen=2000)
        self.force_publisher = self.node.advertise("/world/jev/wrench/persistent", EntityWrench)
        self.node.subscribe(Clock, "/world/jev/clock", self._clock)
        self.node.subscribe(Pose_V, "/world/jev/pose/info", self._pose)
        for name in CAMERAS:
            self.node.subscribe(Image, "/jev/camera/"+name+"/image", partial(self._image, name, "rgb"))
            self.node.subscribe(Image, "/jev/camera/"+name+"/depth_image", partial(self._image, name, "depth"))
        for link in ["base_link"]+[f"rotor_{i}" for i in range(4)]:
            self.node.subscribe(Contacts, "/jev/contacts/"+link, self._contact)

    def _clock(self, msg):
        self.time = msg.sim.sec+msg.sim.nsec*1e-9
        if self.shared_time is not None:
            self.shared_time.value = self.time

    def _pose(self, msg):
        for pose in msg.pose:
            if pose.name.startswith(("crossing_actor_", "red_marker_")):
                self.entity_truth[pose.name] = np.array([pose.position.x, pose.position.y, pose.position.z])
            if pose.name == "drone":
                self.truth = (stamp(msg), np.array([pose.position.x, pose.position.y, pose.position.z]))
                q = pose.orientation
                self.truth_yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

    def _image(self, name, kind, msg):
        if kind == "depth":
            image = np.frombuffer(msg.data, dtype="<f4").reshape(msg.height, msg.step//4)[:, :msg.width].copy()
        else:
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width*3]
            image = image.reshape(msg.height, msg.width, 3).copy()
        with self.lock:
            self.images[name][kind].append((stamp(msg), image))

    def _contact(self, msg):
        if msg.contact:
            self.contacts.append({"time": stamp(msg), "pairs": [
                [c.collision1.name, c.collision2.name] for c in msg.contact]})

    def latest_pairs(self, delay):
        pairs = {}
        with self.lock:
            for name, channels in self.images.items():
                for depth_time, depth in reversed(channels["depth"]):
                    if depth_time > self.time-delay:
                        continue
                    rgb = next((rgb for t, rgb in reversed(channels["rgb"]) if abs(t-depth_time) < .001), None)
                    if rgb is not None:
                        pairs[name] = (depth_time, rgb, depth)
                        break
        return pairs

    def apply_force(self, newtons):
        message = EntityWrench()
        message.entity.name = "drone::base_link"
        message.entity.type = Entity.LINK
        message.wrench.force.x = newtons
        if not self.force_publisher.publish(message):
            raise RuntimeError("Gazebo force publisher failed")
