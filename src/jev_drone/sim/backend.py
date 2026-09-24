"""Adapt live PX4/Gazebo to the existing Jev policy and evaluator contract."""
from dataclasses import asdict
import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from jev_drone.perception.sensors import Frame, Sensors, detect_markers, swept_clearance
from jev_drone.perception.sensors import points
from jev_drone.perception.motion import MotionTracker, colored_components
from jev_drone.world.world import DIRECTIONS
from jev_drone.world.layout import Layout
from jev_drone.sim.scene import WIDTH, camera_rig
from jev_drone.sim.flight import body_velocity_enu
from jev_drone.perception.depth import clearance as fused_clearance


class World:
    realtime = True
    lease = .6

    def __init__(self, config, session):
        self.config, self.session = config, session
        self.layout = Layout(session.scene["rooms"], session.scene["doors"], session.scene["room_connections"])
        self.flight, self.transport = session.flight, session.transport
        self.lease = self.flight.lease
        self.start = np.array(session.scene["start"])
        self.targets, self.mission = session.scene["targets"], session.scene["mission"]
        self.epoch = self.transport.time
        self.started_wall = time.monotonic()
        self.status = "running"
        self.camera_target = np.array([1., 0., 0.])
        self.trace, self.contacts, self.commands = [], [], []
        self.min_clearance = math.inf
        self.near_miss_seconds = 0.
        self._velocity = np.zeros(3)
        self.monitoring = True
        self.telemetry_ready = True
        self.telemetry_outages = []
        session.start_mission(self.epoch)
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    @property
    def time(self):
        return self.transport.time-self.epoch

    @property
    def position(self):
        return self.transport.truth[1].copy()

    @property
    def velocity(self):
        return self._velocity.copy()

    @property
    def camera_direction(self):
        return self.flight.rotation[:, 0].copy()

    def pose_estimate(self):
        return self.flight.position.copy()

    def command(self, choice):
        if self.config.front_controls and choice != "brake":
            raise ValueError("Front-camera flight requires explicit body patches")
        velocity = np.zeros(3)
        if choice.startswith("look_"):
            self.camera_target = np.array(DIRECTIONS[choice[5:]], dtype=float)
        elif choice != "brake":
            speed, direction = choice.split("_", 1)
            velocity = np.array(DIRECTIONS[direction])*(.25 if speed == "slow" else self.config.speed)
        receipt = self.flight.set_velocity(velocity)
        self.commands.append({"time": self.time, "choice": choice, "velocity": velocity.tolist(), **receipt})

    def update_controls(self, choice, patch, revision):
        receipt = self.flight.patch_velocity(patch, revision)
        if receipt["applied"]:
            if self.config.front_controls:
                values = receipt["after"]["body_controls"]
                heading = math.atan2(self.camera_direction[1], self.camera_direction[0])
                velocity = body_velocity_enu(values, heading).tolist()
            else:
                velocity = list(receipt["after"]["velocity_mps"].values())
            self.commands.append({"time": self.time, "choice": choice, "velocity": velocity, **receipt})
        return receipt

    def _monitor(self):
        previous = None
        last_clock, last_clock_wall = self.transport.time, time.monotonic()
        while self.monitoring:
            now = time.monotonic()
            if self.transport.time != last_clock:
                last_clock, last_clock_wall = self.transport.time, now
            fresh = now-last_clock_wall <= 1 and now-self.flight.position_at <= .5
            if self.status == "running" and not fresh:
                if self.telemetry_ready:
                    self.telemetry_outages.append({"started_wall": now, "time": self.time})
                    self.command("brake")
                self.telemetry_ready = False
                if now-self.telemetry_outages[-1]["started_wall"] > 3:
                    self.status = "telemetry_lost"
            elif fresh and not self.telemetry_ready:
                self.telemetry_outages[-1]["recovered_wall"] = now
                self.telemetry_ready = True
            if self.status == "running" and now-self.started_wall > self.config.seconds*2+20:
                self.status = "wall_timeout"
                self.command("brake")
            if self.status == "running" and self.session.motion_error:
                self.status = "actor_error"
                self.command("brake")
            truth = self.transport.truth
            if truth and (previous is None or truth[0]-previous[0] >= .095):
                dt = truth[0]-previous[0] if previous else .1
                if previous:
                    self._velocity = (truth[1]-previous[1])/dt
                previous = truth
                # Evaluator-only conservative sphere proxy, not a collision query or sensor.
                clearance = min(float(np.linalg.norm(np.maximum(np.abs(truth[1]-self.transport.entity_truth.get(
                    f"{box['name']}_{box['body']}", box["center"]))-box["half"], 0))-.40)
                                for box in self.session.scene["geometry"])
                self.min_clearance = min(self.min_clearance, clearance)
                self.near_miss_seconds += dt if clearance < .12 else 0
                self.trace.append({"time": self.time, "wall_time": now-self.started_wall,
                                   "position": truth[1].tolist(), "velocity": self.velocity.tolist(),
                                   "estimate": self.pose_estimate().tolist(), "camera": self.camera_direction.tolist(),
                                   "yaw_rate_rps": self.flight.yaw_rate,
                                   "clearance": clearance, "actors": [self.transport.entity_truth.get(
                                       f"crossing_actor_{a['body']}", np.array([a["x"], 4, .9])).tolist()
                                       for a in self.session.scene["actors"]], "mode": self.flight.mode})
                if self.config.case == "moved_target":
                    target = self.targets["red"]
                    actual = self.transport.entity_truth.get(f"red_marker_{target['body']}")
                    if actual is not None:
                        self.targets = {**self.targets, "red": {**target, "position": actual.tolist()}}
            pending = [c for c in list(self.transport.contacts) if c["time"] > self.epoch]
            if pending:
                self.contacts = [{**c, "time": c["time"]-self.epoch} for c in pending]
                if self.status == "running":
                    self.status = "collision"
                    self.command("brake")
            if self.status == "running" and (not self.flight.armed or self.flight.mode != "OFFBOARD"):
                self.status = "offboard_lost"
                self.command("brake")
            time.sleep(.01)

    def metadata(self):
        return {**self.session.scene, "config": asdict(self.config),
                "physics": "Gazebo Harmonic DART, PX4 1.17 X500 motors and EKF2, free-running 250 Hz target",
                "clearance_metric": "evaluator-only 0.40 m sphere-to-scene AABB proxy; actual contacts recorded separately",
                "camera_control": ("front RGB-D for objects; rear/left/right/up/down depth only; physical body yaw and forward/right/up velocity"
                    if self.config.front_controls else "six fixed body cameras; look actions select attention while braking, yaw held east"),
                "mission_start_sim_time": self.epoch}

    def close(self):
        self.command("brake")
        self.monitoring = False
        self.thread.join(timeout=2)


class CameraSensors:
    save_frame = Sensors.save_frame

    def __init__(self, world):
        self.world = world
        self.config = world.config
        self.cameras, self.hfov = camera_rig(self.config)
        self.rng = np.random.default_rng(self.config.seed+40000)
        self.current = []
        self.blackout_frames = 0
        self.cached = {}
        self.pose_sync_rejections = 0
        self.tracker = MotionTracker(self.config.prediction, body_radius=.40) if self.config.tracking else None
        self.tracker_stamp = -math.inf

    def advance(self, duration):
        # Gazebo, the PX4 controller and camera streams keep running independently.
        time.sleep(max(0, duration))

    def observed_gap_at(self, position):
        if not len(self.observed_cloud):
            return None
        return round(float(np.min(np.linalg.norm(self.observed_cloud-position, axis=1)))-.40, 3)

    def _frames(self):
        world = self.world
        history = list(world.flight.history)
        frames = []
        for name, (stamp, rgb, raw_depth) in world.transport.latest_pairs(self.config.sensor_delay).items():
            if name not in self.cameras:
                continue
            if world.transport.time-stamp > .65 or not history:
                continue
            nearest = min(history, key=lambda p: abs(p[0]-stamp))
            if abs(nearest[0]-stamp) > .06:
                self.pose_sync_rejections += 1
                continue
            key = name, stamp
            if key in self.cached:
                frames.append(self.cached[key])
                continue
            _, pose, rotation = nearest
            offset, angles = self.cameras[name]
            axes = rotation @ Rotation.from_euler("xyz", angles).as_matrix()
            depth = raw_depth.copy()
            depth[~np.isfinite(depth) | (depth < .08) | (depth >= 12)] = np.nan
            depth += self.rng.normal(0, self.config.depth_noise, depth.shape)*(1+depth*depth*.1)
            depth[self.rng.random(depth.shape) < self.config.dropout] = np.nan
            depth = np.round(depth/.02)*.02
            if self.config.case == "blackout" and 12 <= (stamp-world.epoch) % 35 < 17:
                depth[:] = np.nan
                rgb = np.zeros_like(rgb)
                self.blackout_frames += 1
            if self.config.case == "depth_holes":
                depth[:, depth.shape[1]//3:2*depth.shape[1]//3] = np.nan
            frame = Frame(stamp-world.epoch, rgb, depth, pose+rotation@offset,
                          axes[:, 0], -axes[:, 1], axes[:, 2], WIDTH/(2*math.tan(self.hfov/2)), name)
            self.cached[key] = frame
            frames.append(frame)
        self.cached = {k: v for k, v in self.cached.items() if world.transport.time-k[1] < 1}
        self.current = frames
        return frames

    def observation(self):
        world = self.world
        frames = self._frames()
        pose = world.pose_estimate()
        clearance = dict.fromkeys(DIRECTIONS)
        detections = []
        for frame in frames:
            if not self.config.front_controls or frame.camera_name == "east":
                detections.extend(detect_markers(frame))
            for name, direction in DIRECTIONS.items():
                value = swept_clearance(frame, pose, direction, radius=.40)
                if value is not None:
                    clearance[name] = max(clearance[name] or 0., value)
        observation = {"position_estimate": np.round(pose, 3).tolist(),
                "velocity_estimate": np.round(world.flight.velocity, 3).tolist(),
                "camera_direction": np.round(world.camera_direction, 3).tolist(),
                "camera_target": world.camera_target.tolist(), "camera_turning": False,
                "frame_age_seconds": max((world.time-f.time for f in frames), default=None),
                "valid_depth_fraction": float(np.mean([np.isfinite(f.depth).mean() for f in frames])) if frames else 0.,
                "clearance_m": clearance, "detections": detections,
                "position_uncertainty_m": None, "camera_count": len(frames),
                "sensor_contract": "Six fixed wide-angle RGB-D cameras. look commands brake and select attention; all cameras are already visible. Position is EKF2 with simulated external odometry."}
        if self.config.front_controls:
            front = next((frame for frame in frames if frame.camera_name == 'east'), None)
            heading = math.atan2(world.camera_direction[1], world.camera_direction[0])
            forward = np.array([math.cos(heading), math.sin(heading), 0.])
            right = np.array([math.sin(heading), -math.cos(heading), 0.])
            observation.update(yaw_degrees=round(math.degrees(heading), 2),
                yaw_rate_rps=round(world.flight.yaw_rate, 3),
                camera_turning=abs(world.flight.yaw_rate) > .1,
                body_velocity_mps={"forward": round(float(world.flight.velocity@forward), 3),
                                   "right": round(float(world.flight.velocity@right), 3),
                                   "up": round(float(world.flight.velocity[2]), 3)},
                body_clearance_m={}, front_frame_available=any(f.camera_name == "east" for f in frames),
                front_frame_time=front.time if front is not None else None,
                front_valid_depth_fraction=float(np.isfinite(front.depth).mean()) if front is not None else 0.,
                semantic_camera_names=sorted({f.camera_name for f in frames if f.camera_name == "east"}),
                sensor_contract="Front 137.5-degree RGB-D only for object detections. Rear/left/right/up/down depth supplies clearance only. Real body yaw turns the front camera. EKF2 uses simulated external odometry.")
            observation.pop("camera_target")  # Yaw rate control has no fixed look-at target.
            clouds = []
            for frame in frames:
                cloud = points(frame)
                finite = np.isfinite(cloud).all(axis=2)
                if finite.any():
                    clouds.append(cloud[finite])
            self.observed_cloud = np.concatenate(clouds) if clouds else np.empty((0, 3))
            observation["nearest_observed_body_gap_m"] = self.observed_gap_at(pose)
            for name, direction in {"forward": forward, "backward": -forward,
                                    "right": right, "left": -right,
                                    "up": np.array([0., 0., 1.]), "down": np.array([0., 0., -1.])}.items():
                observation["body_clearance_m"][name] = fused_clearance(frames, pose, direction)
        if self.tracker:
            stamp = max((f.time for f in frames), default=-math.inf)
            if stamp > self.tracker_stamp:
                components = [c for f in frames for c in colored_components(f, points(f))]
                self.tracker.update(stamp, components)
                self.tracker_stamp = stamp
            observation.update(self.tracker.observation(pose, world.flight.velocity, self.config.speed, world.time))
        return observation
