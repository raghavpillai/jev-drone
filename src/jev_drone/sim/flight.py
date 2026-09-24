"""MAVLink transport, estimator telemetry, and independent leased setpoint stream."""
from collections import deque
import math
import threading
import time

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

from jev_drone.control.controls import ControlLatch, BODY_CHANNELS


def enu_to_ned(vector):
    x, y, z = vector
    return np.array([y, x, -z], dtype=float)


def attitude_enu(roll, pitch, yaw):
    # MAVLink attitude: body FRD -> NED. Cameras are body FLU -> ENU.
    return np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) @ Rotation.from_euler(
        "xyz", [roll, pitch, yaw]).as_matrix() @ np.diag([1, -1, -1])


def body_velocity_enu(controls, yaw):
    forward, right = controls["forward_mps"], controls.get("right_mps", 0.)
    return np.array([forward*math.cos(yaw)+right*math.sin(yaw),
                     forward*math.sin(yaw)-right*math.cos(yaw), controls["up_mps"]])


class Flight:
    lease = .6

    def __init__(self, clock, *, body_controls=True, lease=.6):
        self.clock = clock
        self.lease = lease
        self.body_controls = body_controls
        self.connection = mavutil.mavlink_connection("udpin:0.0.0.0:14540", source_system=245)
        self.lock = threading.Lock()
        self.position = None
        self.velocity = np.zeros(3)
        self.rotation = np.eye(3)
        self.yaw_rate = 0.
        self.history = deque(maxlen=400)
        self.acks = deque(maxlen=50)
        self.messages = deque(maxlen=100)
        self.armed = False
        self.mode = None
        self.received_at = 0.
        self.position_at = 0.
        self.running = True
        self.active = False
        self.controls = (ControlLatch(self.lease, axes=BODY_CHANNELS, state_key="body_controls")
                         if body_controls else ControlLatch(self.lease))
        self.target_position = None
        self.expirations = []
        self.sent = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        next_send = next_heartbeat = 0.
        while self.running:
            now = time.monotonic()
            for _ in range(200):
                message = self.connection.recv_match(blocking=False)
                if message is None:
                    break
                if message.get_srcSystem() != 1:
                    continue
                kind = message.get_type()
                self.received_at = now
                if kind == "HEARTBEAT":
                    self.armed = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    # pymavlink 2.4.49 interprets Offboard as a submode; PX4 1.17
                    # encodes it as main mode 6 (px4_custom_mode.h).
                    self.mode = ("OFFBOARD" if message.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4
                                 and (message.custom_mode >> 16) & 0xff == 6
                                 else mavutil.mode_string_v10(message))
                elif kind == "LOCAL_POSITION_NED":
                    self.position = enu_to_ned([message.x, message.y, message.z])
                    self.velocity = enu_to_ned([message.vx, message.vy, message.vz])
                    self.position_at = now
                    self.history.append((self.clock(), self.position.copy(), self.rotation.copy()))
                elif kind == "ATTITUDE":
                    self.rotation = attitude_enu(message.roll, message.pitch, message.yaw)
                    self.yaw_rate = -message.yawspeed
                elif kind == "COMMAND_ACK":
                    self.acks.append((now, message.command, message.result))
                elif kind == "STATUSTEXT":
                    self.messages.append(message.text)
            if now >= next_heartbeat and self.connection.target_system:
                with self.lock:
                    # The local experiment harness is also the simulated ground station.
                    self.connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, mavutil.mavlink.MAV_STATE_ACTIVE)
                next_heartbeat = now+1
            if self.active and now >= next_send:
                with self.lock:
                    now = time.monotonic()
                    self._expire_controls(now)
                    expired = self.controls.expires is None
                    values = dict(self.controls.values)
                    heading = math.atan2(self.rotation[1, 0], self.rotation[0, 0])
                    body = self.body_controls and self.target_position is None
                    velocity = (body_velocity_enu(values, heading) if body else
                                np.zeros(3) if self.target_position is not None else np.array(list(values.values())))
                    position = self.target_position
                    revision = self.controls.revision
                    # Calibration position mode is explicit; Jev flights use velocity only.
                    mask = 2552 if position is not None else (1479 if body else 2503)
                    p = enu_to_ned(position) if position is not None else np.zeros(3)
                    v = np.array([values["forward_mps"], values["right_mps"], -values["up_mps"]]) if body else enu_to_ned(velocity)
                    frame = mavutil.mavlink.MAV_FRAME_BODY_NED if body else mavutil.mavlink.MAV_FRAME_LOCAL_NED
                    yaw_rate_ned = -values["yaw_rate_rps"] if body else 0.
                    self.connection.mav.set_position_target_local_ned_send(
                        int(self.clock()*1000) & 0xffffffff, 1, 1,
                        frame, mask, *p, *v, 0, 0, 0, math.pi/2, yaw_rate_ned)
                self.sent.append({"wall": now, "sim": self.clock(), "velocity": velocity.tolist(),
                                  "position": position.tolist() if position is not None else None,
                                  "expired": expired, "revision": revision,
                                  "frame": frame, "mask": mask, "wire_velocity": v.tolist(),
                                  "yaw_rate_ned": yaw_rate_ned, "yaw_enu": heading,
                                  "body_controls": values if body else None})
                next_send = now+.05
            time.sleep(.005)

    def wait(self, predicate, timeout, description):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        raise TimeoutError(description+"; PX4: "+" | ".join(list(self.messages)[-8:]))

    def command_long(self, command, *params):
        started = time.monotonic()
        with self.lock:
            self.connection.mav.command_long_send(1, 1, command, 0, *(list(params)+[0]*(7-len(params))))
        self.wait(lambda: any(t >= started and c == command for t, c, _ in self.acks), 4, "MAVLink ACK timeout")
        result = next(r for t, c, r in reversed(self.acks) if t >= started and c == command)
        if result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(f"PX4 rejected command {command}: {result}; {list(self.messages)[-8:]}")

    def connect(self):
        self.wait(lambda: self.position is not None and self.mode is not None, 60, "PX4 estimator not ready")
        for message_id in (32, 30):
            self.command_long(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, message_id, 20000)
        # The first local-position packet can precede EKF alignment and health checks.
        time.sleep(6)
        self.wait(lambda: np.linalg.norm(self.velocity) < .1, 15, "Estimator did not settle on the ground")

    def takeoff(self, target):
        self.target_position = np.array(target, dtype=float)
        self.active = True
        time.sleep(1.5)
        self.command_long(mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1, 6)
        self.command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        self.wait(lambda: self.armed, 5, "PX4 did not arm")
        self.wait(lambda: np.linalg.norm(self.position-target) < .15 and np.linalg.norm(self.velocity) < .12,
                  30, "Takeoff did not settle")
        time.sleep(4)
        self.set_velocity(np.zeros(3))
        time.sleep(3)

    def set_velocity(self, velocity):
        with self.lock:
            now = time.monotonic()
            self._expire_controls(now)
            self.target_position = None
            if len(velocity) != 3:
                raise ValueError("A full velocity command requires three components")
            if getattr(self, "body_controls", False) and np.any(velocity):
                raise ValueError("Body mode requires named body controls; only a zero world command is allowed")
            patch = (dict.fromkeys(self.controls.axes, 0.) if getattr(self, "body_controls", False)
                     else dict(zip(self.controls.axes, map(float, velocity))))
            return self.controls.update(patch, now)

    def _expire_controls(self, now):
        event = self.controls.expire(now, now-self.position_at > .5)
        if event:
            self.expirations.append({"wall": now, "sim": self.clock(), **event})

    def control_snapshot(self):
        with self.lock:
            self._expire_controls(time.monotonic())
            return self.controls.snapshot()

    def patch_velocity(self, patch, expected_revision):
        with self.lock:
            now = time.monotonic()
            self._expire_controls(now)
            if now-self.position_at > .5:
                snapshot = self.controls.snapshot()
                return {"applied": False, "patch": dict(patch), "expected_revision": expected_revision,
                        "before": snapshot, "after": snapshot, "wall": now, "reason": "telemetry_stale"}
            receipt = self.controls.update(patch, now, expected_revision)
            if receipt["applied"]:
                self.target_position = None
            return receipt

    def close(self):
        self.set_velocity(np.zeros(3))
        self.running = False
        self.thread.join(timeout=2)
        self.connection.close()
