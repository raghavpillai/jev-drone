"""RGB/depth-only frontend. No body IDs, collision queries or object poses.

Colored inspection markers are detected from connected RGB components. Depth
comes from the renderer's depth buffer with explicit defects; it is not learned
monocular depth. Geometry-derived collision distances stay in the evaluator.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import pybullet as p
from scipy.ndimage import label

from jev_drone.world.world import DIRECTIONS
from jev_drone.perception.motion import MotionTracker, colored_components


@dataclass
class Frame:
    time: float
    rgb: np.ndarray
    depth: np.ndarray
    origin: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    focal: float
    camera_name: str | None = None


def basis(direction):
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.array([0., 0., 1.]) if abs(forward[2]) < .95 else np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return forward, right, np.cross(right, forward)


def points(frame):
    height, width = frame.depth.shape
    row, col = np.indices((height, width))
    rays = (frame.forward + ((col-width/2)/frame.focal)[..., None]*frame.right
            - ((row-height/2)/frame.focal)[..., None]*frame.up)
    return frame.origin + frame.depth[..., None]*rays


def detect_markers(frame):
    """Color-coded lab targets only. Identity is decoded from pixels, not IDs."""
    rgb = frame.rgb.astype(float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    masks = {"red": (r > 2*g) & (r > 2*b) & (r > 50),
             "blue": (b > 1.8*r) & (b > 1.8*g) & (b > 50),
             "yellow": (r > 2*b) & (g > 2*b) & (r > 50) & (g > 50) & (r < 1.25*g) & (g < 1.25*r)}
    cloud = points(frame)
    detections = []
    for name, mask in masks.items():
        components, count = label(mask)
        for component in range(1, count+1):
            pixels = (components == component) & np.isfinite(frame.depth)
            if pixels.sum() < 6:
                continue
            detections.append({"marker": name, "position": np.median(cloud[pixels], axis=0).tolist(),
                               "pixels": int(pixels.sum()), "time": frame.time})
    return detections


def swept_clearance(frame, origin, direction, radius=.30):
    """Conservative sampled free tube, reconstructed from camera depth alone.

    Test nine footprint rays at each distance; an occluded or off-image sample
    is unobserved, not free. This is a finite-resolution estimate, not a proof
    that a thin wire or transparent obstacle is absent.
    """
    direction, a, b = basis(direction)
    if np.dot(frame.forward, direction) < .91:
        return None
    height, width = frame.depth.shape
    last = 0.
    for distance in (.2, .4, .6, .9, 1.3, 1.8, 2.5, 3.5):
        samples = np.array([origin + direction*(distance+radius) + radius*(x*a+y*b)
                            for x in (-1, 0, 1) for y in (-1, 0, 1)])
        relative = samples-frame.origin
        z = relative @ frame.forward
        u = width/2 + frame.focal*(relative @ frame.right)/z
        v = height/2 - frame.focal*(relative @ frame.up)/z
        for x, y, axial_depth in zip(u, v, z):
            ix, iy = int(round(x)), int(round(y))
            if axial_depth <= .06 or ix < 1 or ix >= width-1 or iy < 1 or iy >= height-1:
                return last if last else None
            patch = frame.depth[iy-1:iy+2, ix-1:ix+2]
            finite = patch[np.isfinite(patch)]
            if len(finite) < 3:
                return last if last else None
            if np.quantile(finite, .2) < axial_depth+.035:
                return last
        last = distance
    return last


class Sensors:
    width, height = 96, 72
    fov = 80.
    near, far = .06, 12.

    def __init__(self, world):
        self.world = world
        self.config = world.config
        self.rng = np.random.default_rng(self.config.seed+40000)
        self.queue = deque()
        self.current = []
        self.next_capture = 0.
        self.sequence = 0
        self.blackout_frames = 0
        self.render_seconds = 0.
        self.tracker = MotionTracker(self.config.prediction) if self.config.tracking else None
        self.capture()

    def capture(self):
        world = self.world
        config = self.config
        directions = list(DIRECTIONS.values()) if config.rig == "surround" else [world.camera_direction.copy()]
        frames = []
        for direction in directions:
            forward, right, up = basis(direction)
            view = p.computeViewMatrix(world.position, world.position+forward, up)
            projection = p.computeProjectionMatrixFOV(self.fov, self.width/self.height, self.near, self.far)
            image = p.getCameraImage(self.width, self.height, view, projection,
                                     renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK,
                                     physicsClientId=world.client)
            rgb = np.asarray(image[2], dtype=np.uint8).reshape(self.height, self.width, 4)[..., :3].copy()
            buffer = np.asarray(image[3]).reshape(self.height, self.width)
            depth = self.far*self.near/(self.far-(self.far-self.near)*buffer)
            depth = depth + self.rng.normal(0, config.depth_noise, depth.shape)*(1+depth*depth*.1)
            depth = np.round(depth/.02)*.02
            depth[(buffer > .9999) | (depth < .1) | (self.rng.random(depth.shape) < config.dropout)] = np.nan
            # Explicit stress injections, not an optical simulation of glass/lighting.
            if config.case == "blackout" and 12 <= world.time % 35 < 17:
                depth[:] = np.nan
                rgb[:] = 0
                self.blackout_frames += 1
            if config.case == "depth_holes":
                depth[:, self.width//3:2*self.width//3] = np.nan
            focal = self.height/(2*math.tan(math.radians(self.fov/2)))
            frames.append(Frame(world.time, rgb, depth.astype(np.float32), world.pose_estimate(),
                                forward.copy(), right, up, focal))
        self.queue.append(frames)
        self.sequence += 1
        self.next_capture = world.time+1/config.sensor_hz

    def advance(self, duration):
        end = self.world.time+duration
        while self.world.time < end-1e-9 and self.world.status == "running":
            step = min(end-self.world.time, max(1e-6, self.next_capture-self.world.time))
            self.world.advance(step)
            if self.world.time >= self.next_capture-1e-8:
                self.capture()
        self.deliver()

    def deliver(self):
        while self.queue and self.queue[0][0].time <= self.world.time-self.config.sensor_delay+1e-8:
            self.current = self.queue.popleft()
            if self.tracker:
                detections = []
                for frame in self.current:
                    detections.extend(colored_components(frame, points(frame)))
                self.tracker.update(self.current[0].time, detections)
        return self.current

    def observation(self):
        frames = self.deliver()
        pose = self.world.pose_estimate()
        detections = []
        clearance = dict.fromkeys(DIRECTIONS)
        age = None
        valid = 0.
        for frame in frames:
            age = self.world.time-frame.time
            if age > .65:
                continue
            valid += float(np.isfinite(frame.depth).mean())/len(frames)
            detections.extend(detect_markers(frame))
            for name, direction in DIRECTIONS.items():
                estimate = swept_clearance(frame, pose, direction)
                if estimate is not None:
                    clearance[name] = round(estimate, 2)
        observation = {"position_estimate": np.round(pose, 2).tolist(),
                "velocity_estimate": np.round(self.world.velocity, 2).tolist(),
                "camera_direction": np.round(self.world.camera_direction, 2).tolist(),
                "camera_target": self.world.camera_target.tolist(),
                "camera_turning": bool(np.dot(self.world.camera_direction, self.world.camera_target) < .995),
                "frame_age_seconds": round(age, 3) if age is not None else None,
                "valid_depth_fraction": round(valid, 3), "clearance_m": clearance,
                "detections": detections,
                "position_uncertainty_m": round(.03+self.config.odometry_drift*self.world.time, 2)}
        if self.tracker:
            observation.update(self.tracker.observation(pose, self.world.velocity, self.config.speed, self.world.time))
        return observation

    def save_frame(self, path):
        if not self.current:
            return
        np.savez_compressed(path, rgb=np.stack([f.rgb for f in self.current]),
                            depth=np.stack([f.depth for f in self.current]),
                            origins=np.array([f.origin for f in self.current]),
                            forward=np.array([f.forward for f in self.current]),
                            right=np.array([f.right for f in self.current]),
                            up=np.array([f.up for f in self.current]),
                            focal=np.array([f.focal for f in self.current]),
                            camera_names=np.array([f.camera_name or "" for f in self.current]),
                            times=np.array([f.time for f in self.current]))
