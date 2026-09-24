"""PyBullet indoor world with a force-controlled, stabilized drone surrogate.

This is deliberately not a rotor/aerodynamics simulator. Navigation commands go
through a bounded acceleration servo, gravity compensation, drag and wind. The
collision body is a 22 cm radius sphere. Camera rendering never uses segmentation.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import pybullet as p

from jev_drone.world.layout import SMALL, for_case

DIRECTIONS = {
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
    "north": (0, 1, 0),
    "south": (0, -1, 0),
    "up": (0, 0, 1),
    "down": (0, 0, -1),
}
ROOMS, DOORS = SMALL.rooms, SMALL.doors
COLORS = {
    "red": (0.95, 0.06, 0.06, 1),
    "blue": (0.06, 0.12, 0.95, 1),
    "yellow": (0.95, 0.9, 0.04, 1),
}


@dataclass
class Config:
    seed: int = 2101
    case: str = "clutter"
    rig: str = "gimbal"
    memory: bool = True
    sensor_hz: float = 8.0
    sensor_delay: float = 0.10
    depth_noise: float = 0.015
    dropout: float = 0.06
    odometry_drift: float = 0.0015
    wind: float = 0.12
    speed: float = 0.6
    seconds: float = 150.0
    budget: float = 0.18
    policy: str = "v1"
    tracking: bool = False
    prediction: str = "linear"
    servo: str = "p"


class World:
    dt = 1 / 120
    radius = 0.22
    camera_rate = math.radians(120)
    lease = 0.6

    def __init__(self, config):
        self.config = config
        self.layout = for_case(config.case)
        self.rng = np.random.default_rng(config.seed)
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=30, physicsClientId=self.client)
        self.time = 0.0
        self.status = "running"
        self.command_velocity = np.zeros(3)
        self.command_until = 0.0
        self.velocity_integral = np.zeros(3)
        self.camera_direction = np.array([1.0, 0.0, 0.0])
        self.camera_target = self.camera_direction.copy()
        self.geometry = []
        self.bodies = []
        self.dynamic = []
        self.targets = {}
        self.start = np.array([1.4, 2.0, 1.3])
        self.trace = []
        self.contacts = []
        self.min_clearance = 100.0
        self.near_miss_seconds = 0.0
        self._build()
        shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=self.radius, physicsClientId=self.client
        )
        self.drone = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=shape,
            basePosition=self.start,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.drone, -1, linearDamping=0.0, angularDamping=0.95, physicsClientId=self.client
        )
        self.drift_direction = self.rng.normal(size=3)
        self.drift_direction /= np.linalg.norm(self.drift_direction)
        self._record()

    def box(self, name, center, half, color=(0.58, 0.57, 0.55, 1), dynamic=False):
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client
        )
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=center,
            physicsClientId=self.client,
        )
        self.bodies.append(body)
        self.geometry.append(
            {
                "name": name,
                "center": list(center),
                "half": list(half),
                "color": list(color),
                "body": body,
                "dynamic": dynamic,
            }
        )
        return body

    def _wall_with_door(self, axis, coordinate, lo, hi, doors, width):
        height = float(self.layout.bounds[1][2])
        door_height = 2.8 if self.layout is SMALL else 2.4
        edges = [lo]
        for door in sorted(doors):
            edges += [door - width / 2, door + width / 2]
        edges.append(hi)
        for a, b in zip(edges[::2], edges[1::2]):
            if axis == 0:
                self.box(
                    "wall",
                    (coordinate, (a + b) / 2, height / 2),
                    (0.075, (b - a) / 2, height / 2),
                    (0.8, 0.81, 0.84, 1),
                )
            else:
                self.box(
                    "wall",
                    ((a + b) / 2, coordinate, height / 2),
                    ((b - a) / 2, 0.075, height / 2),
                    (0.8, 0.81, 0.84, 1),
                )
        for door in doors:
            z, h = (height + door_height) / 2, (height - door_height) / 2
            center = (coordinate, door, z) if axis == 0 else (door, coordinate, z)
            half = (0.075, width / 2, h) if axis == 0 else (width / 2, 0.075, h)
            self.box("lintel", center, half)

    def _build(self):
        case = self.config.case
        if self.layout is not SMALL:
            from jev_drone.world.house_scene import build

            build(self)
            return
        for name, center, half in [
            ("floor", (6, 4, -0.1), (6, 4, 0.1)),
            ("ceiling", (6, 4, 4.1), (6, 4, 0.1)),
            ("west_wall", (-0.1, 4, 2), (0.1, 4, 2)),
            ("east_wall", (12.1, 4, 2), (0.1, 4, 2)),
            ("south_wall", (6, -0.1, 2), (6, 0.1, 2)),
            ("north_wall", (6, 8.1, 2), (6, 0.1, 2)),
        ]:
            self.box(name, center, half, (0.78, 0.8, 0.82, 1))
        self._wall_with_door(0, 6, 0, 8, [2, 6], 0.95 if case == "narrow" else 1.65)
        self._wall_with_door(1, 4, 6, 12, [9], 1.65)
        jitter = self.rng.uniform(-0.25, 0.25, 2)
        # Furniture is assembled from surfaces and legs, not floor-solid boxes.
        self.box(
            "table_top", (3.5 + jitter[0], 4.2, 1.05), (0.85, 0.7, 0.07), (0.45, 0.27, 0.12, 1)
        )
        for x in (2.8 + jitter[0], 4.2 + jitter[0]):
            for y in (3.65, 4.75):
                self.box("table_leg", (x, y, 0.5), (0.055, 0.055, 0.5), (0.3, 0.2, 0.1, 1))
        self.box("bed", (9.5, 1.1, 0.4), (1.0, 0.7, 0.4), (0.55, 0.36, 0.34, 1))
        self.box("chair_seat", (7.4, 6.8, 0.6), (0.35, 0.35, 0.055), (0.4, 0.3, 0.2, 1))
        self.box("chair_back", (7.4, 7.12, 1), (0.35, 0.045, 0.4), (0.4, 0.3, 0.2, 1))
        for x in (7.12, 7.68):
            for y in (6.52, 7.08):
                self.box("chair_leg", (x, y, 0.28), (0.04, 0.04, 0.28))
        for z in (0.2, 1.2, 2.2):
            self.box("shelf", (11.55, 6.1, z), (0.35, 1.1, 0.06), (0.4, 0.25, 0.12, 1))
        for y in (5.0, 7.2):
            self.box("shelf_side", (11.55, y, 1.2), (0.35, 0.06, 1.2), (0.4, 0.25, 0.12, 1))
        self.box("thin_pole", (4.65, 2.8, 1.25), (0.045, 0.045, 1.25))
        if case in ("overhang", "vertical"):
            self.box("hanging_beam", (3.6, 2, 1.65), (0.2, 1.0, 0.28))
            self.box("low_screen", (4.6, 2, 0.55), (0.14, 1.25, 0.55))
        if case in ("occluded", "moved_target"):
            self.box("search_screen", (9.6, 2.55, 1.15), (0.12, 0.85, 1.15))
        if case == "blocked_door":
            self.box(
                "unexpected_closed_door", (6, 2, 1.35), (0.08, 0.825, 1.35), (0.4, 0.28, 0.16, 1)
            )
        if case in ("crossing", "crowded"):
            for index, x in enumerate([3.7] if case == "crossing" else [3.7, 7.7, 9.8]):
                body = self.box(
                    "crossing_actor", (x, 4, 0.9), (0.22, 0.22, 0.9), (0.18, 0.6, 0.28, 1), True
                )
                self.dynamic.append(
                    {
                        "body": body,
                        "x": x,
                        "phase": float(self.rng.uniform(0, 2 * math.pi)),
                        "index": index,
                    }
                )
        target_positions = {
            "red": [10.7, 2.8 + jitter[1], 1.5],
            "blue": [10.7, 6.6, 1.5],
            "yellow": [2.5, 6.6, 1.5],
        }
        if case == "vertical":
            target_positions["red"] = [10.6, 2.8, 3.1]
        for name, point in target_positions.items():
            body = self.box(
                name + "_marker",
                point,
                (0.20, 0.20, 0.20),
                COLORS[name],
                case == "moved_target" and name == "red",
            )
            self.targets[name] = {"position": point, "body": body}
        self.mission = (
            [
                {"marker": "red", "room": "bedroom"},
                {"marker": "blue", "room": "study"},
                {"marker": "yellow", "room": "living"},
                {"dock": self.start.tolist()},
            ]
            if case in ("sequence", "crowded")
            else [{"marker": "red", "room": "bedroom"}, {"dock": self.start.tolist()}]
        )

    @property
    def position(self):
        return np.array(p.getBasePositionAndOrientation(self.drone, physicsClientId=self.client)[0])

    @property
    def velocity(self):
        return np.array(p.getBaseVelocity(self.drone, physicsClientId=self.client)[0])

    def pose_estimate(self):
        # A bounded-rate drift surrogate; not a visual-inertial estimator.
        drift = self.drift_direction * self.config.odometry_drift * self.time
        return self.position + drift

    def command(self, choice):
        if choice.startswith("look_"):
            self.camera_target = np.array(DIRECTIONS[choice[5:]], dtype=float)
            self.command_velocity = np.zeros(3)
        elif choice == "brake":
            self.command_velocity = np.zeros(3)
        else:
            speed, direction = choice.split("_", 1)
            magnitude = 0.25 if speed == "slow" else self.config.speed
            self.command_velocity = np.array(DIRECTIONS[direction]) * magnitude
        self.command_until = self.time + self.lease

    def advance(self, duration):
        count = max(1, math.ceil(duration / self.dt))
        dt = duration / count
        if dt <= 0:
            return
        p.setTimeStep(dt, physicsClientId=self.client)
        for _ in range(count):
            if self.status != "running":
                break
            desired = self.command_velocity if self.time < self.command_until else np.zeros(3)
            error = desired - self.velocity
            acceleration = error / 0.28
            if self.config.servo == "pi":
                self.velocity_integral = np.clip(self.velocity_integral + error * dt, -0.5, 0.5)
                acceleration += 4.0 * self.velocity_integral
            norm = np.linalg.norm(acceleration)
            if norm > 1.8:
                acceleration *= 1.8 / norm
            wind = self.config.wind * np.array(
                [math.sin(0.7 * self.time), math.cos(0.43 * self.time), 0.2 * math.sin(self.time)]
            )
            force = acceleration + np.array([0, 0, 9.81]) + wind - 0.1 * self.velocity
            p.applyExternalForce(
                self.drone, -1, force, self.position, p.WORLD_FRAME, physicsClientId=self.client
            )
            self._turn_camera(dt)
            for actor in self.dynamic:
                y = 2 + 1.35 * math.sin(0.65 * self.time + actor["phase"])
                if actor["x"] > 6:
                    y = 5.7 + 1.25 * math.sin(0.65 * self.time + actor["phase"])
                p.resetBasePositionAndOrientation(
                    actor["body"], (actor["x"], y, 0.9), (0, 0, 0, 1), physicsClientId=self.client
                )
            if self.config.case == "moved_target" and self.time >= 18:
                target = self.targets["red"]
                target["position"] = [8.2, 3.1, 1.5]
                p.resetBasePositionAndOrientation(
                    target["body"], target["position"], (0, 0, 0, 1), physicsClientId=self.client
                )
            p.stepSimulation(physicsClientId=self.client)
            self.time += dt
            contact = p.getContactPoints(bodyA=self.drone, physicsClientId=self.client)
            if contact:
                self.status = "collision"
                self.contacts.append(
                    {
                        "time": self.time,
                        "bodies": [c[2] for c in contact],
                        "position": self.position.tolist(),
                    }
                )
            if not self.trace or self.time - self.trace[-1]["time"] >= 0.1 - 1e-8:
                self._record()

    def _turn_camera(self, dt):
        dot = np.clip(np.dot(self.camera_direction, self.camera_target), -1, 1)
        angle = math.acos(dot)
        if angle < 1e-7:
            return
        fraction = min(1.0, self.camera_rate * dt / angle)
        tangent = self.camera_target - dot * self.camera_direction
        if np.linalg.norm(tangent) < 1e-7:
            tangent = np.cross(self.camera_direction, [0, 0, 1])
            if np.linalg.norm(tangent) < 1e-7:
                tangent = np.array([0.0, 1.0, 0.0])
        tangent /= np.linalg.norm(tangent)
        self.camera_direction = (
            math.cos(angle * fraction) * self.camera_direction
            + math.sin(angle * fraction) * tangent
        )

    def _record(self):
        # Ground-truth distances are evaluator-only; never used by control/sensors.
        clearance = 100.0
        for body in self.bodies:
            points = p.getClosestPoints(self.drone, body, 0.5, physicsClientId=self.client)
            if points:
                clearance = min(clearance, min(point[8] for point in points))
        self.min_clearance = min(self.min_clearance, clearance)
        if clearance < 0.12:
            self.near_miss_seconds += 0.1
        self.trace.append(
            {
                "time": round(self.time, 4),
                "position": self.position.tolist(),
                "velocity": self.velocity.tolist(),
                "camera": self.camera_direction.tolist(),
                "clearance": clearance,
                "actors": [
                    list(p.getBasePositionAndOrientation(a["body"], physicsClientId=self.client)[0])
                    for a in self.dynamic
                ],
            }
        )

    def metadata(self):
        return {
            "config": asdict(self.config),
            "geometry": self.geometry,
            "mission": self.mission,
            "targets": self.targets,
            "start": self.start.tolist(),
            "rooms": self.layout.rooms,
            "doors": self.layout.doors,
            "room_connections": self.layout.connections,
            "physics": "PyBullet 120 Hz, force velocity servo, sphere hull; no rotor model",
        }

    def close(self):
        p.disconnect(physicsClientId=self.client)
