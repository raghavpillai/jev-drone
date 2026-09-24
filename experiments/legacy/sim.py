"""Deterministic stabilized translation, range sensing, and swept-box collisions.

The vehicle is a 0.4 m axis-aligned cube with a fixed heading. This deliberately
tests navigation, not rotor dynamics, visual perception, or attitude control.
"""

import math
import random
from dataclasses import asdict, dataclass, field

Vec = tuple[float, float, float]
ACTIONS: dict[str, Vec] = {
    "forward": (1, 0, 0),
    "forward_left": (1, 1, 0),
    "left": (0, 1, 0),
    "back_left": (-1, 1, 0),
    "back": (-1, 0, 0),
    "back_right": (-1, -1, 0),
    "right": (0, -1, 0),
    "forward_right": (1, -1, 0),
    "up": (0, 0, 1),
    "down": (0, 0, -1),
    "brake": (0, 0, 0),
}


def norm(v: Vec) -> float:
    return math.sqrt(sum(x * x for x in v))


def subtract(a: Vec, b: Vec) -> Vec:
    return tuple(x - y for x, y in zip(a, b))


def unit(v: Vec) -> Vec:
    n = norm(v)
    return tuple(x / n for x in v) if n else (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Box:
    lo: Vec
    hi: Vec

    def expanded(self, radius: float) -> "Box":
        return Box(tuple(x - radius for x in self.lo), tuple(x + radius for x in self.hi))


def ray_box(origin: Vec, direction: Vec, box: Box) -> float | None:
    """Distance parameter to a closed box; zero when already inside/on it."""
    enter, leave = -math.inf, math.inf
    for p, d, lo, hi in zip(origin, direction, box.lo, box.hi):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return None
        else:
            a, b = sorted(((lo - p) / d, (hi - p) / d))
            enter, leave = max(enter, a), min(leave, b)
    if leave < max(enter, 0):
        return None
    return max(enter, 0.0)


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    start: Vec
    goal: Vec
    obstacles: tuple[Box, ...]
    size: Vec = (12.0, 10.0, 4.0)


def scenario(name: str, seed: int) -> Scenario:
    rng = random.Random(seed)
    y = 5.0 + rng.uniform(-0.35, 0.35)
    start, goal = (1.0, y, 1.0), (10.5, y, 1.0)
    layouts = {
        "open": (),
        "pillar": (Box((4.5, 4.0, 0.0), (5.5, 6.0, 4.0)),),
        "wall": (Box((5.0, 2.0, 0.0), (5.4, 8.0, 4.0)),),
        "overpass": (Box((5.0, 0.0, 0.0), (6.0, 10.0, 2.4)),),
        "u_trap": (
            Box((7.0, 3.0, 0.0), (7.4, 7.0, 4.0)),
            Box((3.0, 3.0, 0.0), (7.4, 3.4, 4.0)),
            Box((3.0, 6.6, 0.0), (7.4, 7.0, 4.0)),
        ),
    }
    if name == "u_trap":
        start = (5.0, y, 1.0)
    obstacles = layouts[name]
    # Held-out seeds change geometry as well as the starting point.
    if seed:
        shift = rng.uniform(-0.5, 0.5)
        obstacles = tuple(
            Box((b.lo[0] + shift, b.lo[1], b.lo[2]), (b.hi[0] + shift, b.hi[1], b.hi[2]))
            for b in obstacles
        )
        if seed % 2:
            start = (12.0 - start[0], start[1], start[2])
            goal = (12.0 - goal[0], goal[1], goal[2])
            obstacles = tuple(
                Box((12.0 - b.hi[0], b.lo[1], b.lo[2]), (12.0 - b.lo[0], b.hi[1], b.hi[2]))
                for b in obstacles
            )
    return Scenario(name, seed, start, goal, obstacles)


@dataclass
class Simulation:
    world: Scenario
    speed: float = 1.0
    acceleration: float = 2.0
    radius: float = 0.2
    sensor_range: float = 4.0
    goal_radius: float = 0.65
    command_lease: float = 0.8
    position: Vec = field(init=False)
    velocity: Vec = (0.0, 0.0, 0.0)
    time: float = 0.0
    action: str = "brake"
    command_speed: float = field(init=False)
    command_expires: float = 0.0
    status: str = "running"
    settled: float = 0.0
    path_length: float = 0.0
    brake_time: float = 0.0
    min_clearance: float = math.inf
    trace: list = field(default_factory=list)

    def __post_init__(self):
        self.position = self.world.start
        self.command_speed = self.speed
        self.record()

    def clearance(self, direction: Vec) -> float:
        d = unit(direction)
        distance = self.sensor_range
        for p, v, limit in zip(self.position, d, self.world.size):
            if v > 1e-12:
                distance = min(distance, (limit - self.radius - p) / v)
            elif v < -1e-12:
                distance = min(distance, (self.radius - p) / v)
        for box in self.world.obstacles:
            hit = ray_box(self.position, d, box.expanded(self.radius))
            if hit is not None:
                distance = min(distance, hit)
        return max(0.0, distance)

    def command(self, action: str, speed: float | None = None):
        if action not in ACTIONS:
            raise ValueError("Unknown action")
        requested_speed = self.speed if speed is None else speed
        if not math.isfinite(requested_speed) or not 0 <= requested_speed <= self.speed:
            raise ValueError("Speed outside actuator limits")
        self.action = action
        self.command_speed = requested_speed
        self.command_expires = self.time + self.command_lease

    def advance(self, duration: float):
        remaining = duration
        while remaining > 1e-9 and self.status == "running":
            dt = min(0.02, remaining)
            if self.time < self.command_expires < self.time + dt:
                dt = self.command_expires - self.time
            active = self.action if self.time < self.command_expires - 1e-9 else "brake"
            target = tuple(x * self.command_speed for x in unit(ACTIONS[active]))
            delta = subtract(target, self.velocity)
            change = min(norm(delta), self.acceleration * dt)
            new_velocity = tuple(v + d * change for v, d in zip(self.velocity, unit(delta)))
            displacement = tuple((a + b) * 0.5 * dt for a, b in zip(self.velocity, new_velocity))
            distance = norm(displacement)
            free = self.clearance(displacement) if distance else self.sensor_range
            self.min_clearance = min(self.min_clearance, free)
            collision = distance > 0 and distance >= free - 1e-9
            fraction = max(0.0, min(1.0, free / distance)) if collision else 1.0
            self.position = tuple(p + d * fraction for p, d in zip(self.position, displacement))
            self.path_length += distance * fraction
            self.velocity = new_velocity
            self.time += dt
            remaining -= dt
            if active == "brake":
                self.brake_time += dt
            if collision:
                self.status = "collision"
            if (
                norm(subtract(self.world.goal, self.position)) <= self.goal_radius
                and norm(self.velocity) <= 0.15
            ):
                self.settled += dt
                if self.settled >= 0.4:
                    self.status = "success"
            else:
                self.settled = 0.0
            if not self.trace or self.time - self.trace[-1][0] >= 0.099 or self.status != "running":
                self.record()

    def record(self):
        self.trace.append([round(self.time, 4), *self.position, *self.velocity])

    def observation(self, history: list, target: Vec | None = None) -> dict:
        delta = subtract(self.world.goal if target is None else target, self.position)
        labels = []
        for value, positive, negative in zip(
            delta, ["forward", "left", "up"], ["back", "right", "down"]
        ):
            if abs(value) > 0.3:
                labels.append(positive if value > 0 else negative)
        ranges = {}
        for name, direction in ACTIONS.items():
            if name == "brake":
                continue
            distance = self.clearance(direction)
            ranges[name] = {
                "metres": round(distance, 2),
                "proximity": "very_close"
                if distance < 0.65
                else "near"
                if distance < 1.3
                else "open",
            }
        return {
            "mission": "Reach the destination and stop inside its arrival radius.",
            "axes": "Fixed heading: forward=+x, left=+y, up=+z. All directions are relative to this fixed heading.",
            "goal": {
                "direction": "+".join(labels) or "here",
                "offset_metres_xyz": [round(x, 2) for x in delta],
                "distance_metres": round(norm(delta), 2),
                "inside_arrival_radius": norm(delta) <= self.goal_radius,
            },
            "position_metres_xyz": [round(x, 2) for x in self.position],
            "velocity_metres_per_second_xyz": [round(x, 2) for x in self.velocity],
            "ranges": ranges,
            "sensor_description": "Clearances account for vehicle size. Readings cap at 4 m; open is only a local observation, not a promise the route stays clear.",
            "recent": history[-6:],
        }

    def summary(self) -> dict:
        return {
            "world": asdict(self.world),
            "status": self.status,
            "simulation_seconds": round(self.time, 3),
            "path_metres": round(self.path_length, 3),
            "final_distance_metres": round(norm(subtract(self.world.goal, self.position)), 3),
            "braking_seconds": round(self.brake_time, 3),
            "trace": self.trace,
            "physics": {
                "speed_mps": self.speed,
                "acceleration_mps2": self.acceleration,
                "vehicle_half_extent_m": self.radius,
                "goal_radius_m": self.goal_radius,
                "command_lease_s": self.command_lease,
            },
        }
