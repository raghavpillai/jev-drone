"""Experimental RGB-D blob tracking; no simulator identities or actor trajectories.

This controlled renderer has colored objects and neutral walls. The frontend
tracks any sufficiently saturated connected component, including furniture and
markers. It is not a person detector and will not generalize to arbitrary video.
"""

from collections import deque

import numpy as np
from scipy.ndimage import label

from jev_drone.world.world import DIRECTIONS


def colored_components(frame, cloud):
    rgb = frame.rgb.astype(float)
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    delta = maximum - minimum
    visible = (maximum > 50) & (delta > 0.25 * maximum) & np.isfinite(frame.depth)
    denominator = np.maximum(delta, 1)
    hue = np.zeros_like(maximum)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.where(maximum == r, ((g - b) / denominator) % 6, hue)
    hue = np.where(maximum == g, (b - r) / denominator + 2, hue)
    hue = np.where(maximum == b, (r - g) / denominator + 4, hue)
    bins = np.floor(hue + 0.5).astype(int) % 6
    detections = []
    for color in range(6):
        components, count = label(visible & (bins == color))
        for component in range(1, count + 1):
            mask = components == component
            if mask.sum() < 12:
                continue
            measured = cloud[mask]
            position = np.median(measured, axis=0)
            low, high = np.quantile(measured, [0.05, 0.95], axis=0)
            detections.append(
                {
                    "color_bin": color,
                    "position": position,
                    "half_extent": np.clip((high - low) / 2 + 0.12, 0.20, 1.5),
                    "pixels": int(mask.sum()),
                }
            )
    return detections


class MotionTracker:
    def __init__(self, prediction="linear", body_radius=0.30):
        self.tracks = {}
        self.next_id = 1
        self.time = -1.0
        self.prediction = prediction
        self.body_radius = body_radius

    def update(self, time, detections):
        if time <= self.time:
            return
        self.time = time
        detections = sorted(detections, key=lambda d: -d["pixels"])
        merged = []
        for detection in detections:
            if not any(
                detection["color_bin"] == d["color_bin"]
                and np.linalg.norm(detection["position"] - d["position"]) < 0.65
                for d in merged
            ):
                merged.append(detection)
        assigned = set()
        for detection in merged:
            candidates = [
                (np.linalg.norm(track["position"] - detection["position"]), identity)
                for identity, track in self.tracks.items()
                if identity not in assigned
                and track["color_bin"] == detection["color_bin"]
                and time - track["last_seen"] < 0.8
            ]
            best = min(candidates) if candidates else None
            if best and best[0] < 0.7:
                identity = best[1]
                track = self.tracks[identity]
            else:
                identity = self.next_id
                self.next_id += 1
                track = {"history": deque(maxlen=6), "velocity": np.zeros(3), "last_moving": -100.0}
                self.tracks[identity] = track
            assigned.add(identity)
            track.update(detection, last_seen=time)
            track["history"].append((time, detection["position"].copy()))
            if len(track["history"]) >= 4:
                times = np.array([entry[0] for entry in track["history"]])
                positions = np.array([entry[1] for entry in track["history"]])
                times -= times.mean()
                velocity = (times[:, None] * (positions - positions.mean(axis=0))).sum(
                    axis=0
                ) / max(1e-8, (times * times).sum())
                track["velocity"] = velocity if np.linalg.norm(velocity) < 2.5 else np.zeros(3)
                if np.linalg.norm(track["velocity"]) >= 0.18:
                    track["last_moving"] = time
        self.tracks = {
            identity: track
            for identity, track in self.tracks.items()
            if time - track["last_seen"] < 1.2
        }

    def observation(self, position, velocity, speed, now):
        objects = []
        course_risk = False
        risks = {name: False for name in DIRECTIONS}
        for identity, track in self.tracks.items():
            moving = np.linalg.norm(track["velocity"]) >= 0.18
            if self.prediction == "uncertain":
                moving = now - track["last_moving"] < 3.0
            if len(track["history"]) < 4 or not moving:
                continue
            age = now - track["last_seen"]
            center = track["position"] + age * track["velocity"]
            extent = track["half_extent"] + self.body_radius + 0.15 * age

            def risk(drone_velocity):
                for t in np.linspace(0, 1.4, 15):
                    uncertainty = np.zeros(3)
                    displacement = t * drone_velocity
                    if self.prediction == "uncertain":
                        # Possible horizontal acceleration/reversal; no known actor path.
                        uncertainty[:2] = 0.5 * t * t
                        # A command takes effect after perception/API delay and servo lag.
                        after_delay = max(0.0, t - 0.45)
                        displacement = (
                            velocity * min(t, 0.45)
                            + drone_velocity * after_delay
                            + (velocity - drone_velocity) * 0.28 * (1 - np.exp(-after_delay / 0.28))
                        )
                    relative = center + t * track["velocity"] - (position + displacement)
                    if np.all(np.abs(relative) <= extent + uncertainty):
                        return True
                return False

            course_risk |= risk(velocity)
            for name, direction in DIRECTIONS.items():
                risks[name] |= risk(np.array(direction) * speed)
            objects.append(
                {
                    "track": identity,
                    "position_estimate": np.round(center, 2).tolist(),
                    "velocity_estimate": np.round(track["velocity"], 2).tolist(),
                    "observed_half_extent": np.round(track["half_extent"], 2).tolist(),
                    "age_seconds": round(age, 2),
                }
            )
        return {
            "moving_visual_tracks": objects,
            "current_course_risk": bool(course_risk),
            "predicted_risk_by_direction": {k: bool(v) for k, v in risks.items()},
            "prediction_horizon_seconds": 1.4,
            "prediction_method": self.prediction,
        }
