"""Evaluation scenes: known rooms/doors, hidden furniture and optional occlusion."""

import random
from dataclasses import dataclass, replace

from experiments.legacy.search_world import (
    PORTALS,
    ROOMS,
    Home,
    Object,
    make_home,
    search_waypoints,
)
from experiments.legacy.sim import Box


@dataclass(frozen=True)
class Scene:
    home: Home
    portals: list
    points: dict
    mission: str
    scope: tuple[str, ...]


def make_scene(seed, case):
    target_room = {
        "near": "living_room",
        "far": "study",
        "occluded": "bedroom",
        "absent": "bedroom",
    }[case]
    home = make_home(seed, target_room, absent=case == "absent")
    rng = random.Random(seed + 500)
    centers = [rng.choice([4.5, 5.5, 6.5]) for _ in PORTALS]
    portals = []
    for i, (portal, center) in enumerate(zip(PORTALS, centers)):
        x = (i + 1) * 6
        portals.append(
            {
                "rooms": portal["rooms"],
                "opening_xyz": [[x - 0.15, center - 1.5, 0], [x + 0.15, center + 1.5, 4]],
                "approaches": {
                    room: [point[0], center, 1.4] for room, point in portal["approaches"].items()
                },
            }
        )
    walls = tuple(
        Box((x - 0.15, a, 0), (x + 0.15, b, 4))
        for x, center in zip((6, 12), centers)
        for a, b in ((0, center - 1.5), (center + 1.5, 10))
    )
    objects = list(home.objects)
    if case == "occluded":
        objects.append(
            Object(
                "object_screen",
                "screen",
                Box((7.8, 7.7, 0), (10.8, 8.1, 2.15)),
                "A broad opaque upright partition screen that blocks the view behind it.",
            )
        )
    world = replace(home.world, obstacles=walls + tuple(o.box for o in objects))
    points = search_waypoints()
    for i, portal in enumerate(portals):
        for room, point in portal["approaches"].items():
            points[f"door_{i}_{room}"]["xyz"] = point
    mission = "Find a bookcase anywhere in this home. Approach it, stop nearby, and report found. If the search is exhausted, report not found."
    scope = tuple(ROOMS)
    if case == "occluded":
        mission = "Find the bookcase in the bedroom. Approach it, stop nearby, and report found. If the bedroom search is exhausted, report not found."
        scope = ("bedroom",)
    return Scene(Home(world, tuple(objects), home.target_id), portals, points, mission, scope)
