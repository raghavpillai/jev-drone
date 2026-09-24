"""Three known rooms with hidden furniture and visibility-limited object sensing."""

import itertools
import random
from dataclasses import asdict, dataclass

from experiments.legacy.sim import Box, Scenario, norm, ray_box, subtract, unit

ROOMS = {
    name: {"bounds_xyz": [[i * 6, 0, 0], [(i + 1) * 6, 10, 4]]}
    for i, name in enumerate(["living_room", "bedroom", "study"])
}
PORTALS = [
    {
        "rooms": [a, b],
        "opening_xyz": [[x - 0.15, 4, 0], [x + 0.15, 7, 4]],
        "approaches": {a: [x - 1.2, 5.5, 1.4], b: [x + 1.2, 5.5, 1.4]},
    }
    for a, b, x in [("living_room", "bedroom", 6), ("bedroom", "study", 12)]
]


@dataclass(frozen=True)
class Object:
    id: str
    kind: str  # Evaluator-only ground truth, never sent to a model.
    box: Box
    appearance: str


@dataclass(frozen=True)
class Home:
    world: Scenario
    objects: tuple[Object, ...]
    target_id: str | None


def room_at(position):
    for name, room in ROOMS.items():
        if all(a <= p < b for p, a, b in zip(position, *room["bounds_xyz"])):
            return name
    return "outside"


def make_home(seed, target_room="bedroom", absent=False):
    rng = random.Random(seed)
    walls = tuple(
        Box((x - 0.15, y0, 0), (x + 0.15, y1, 4)) for x in (6, 12) for y0, y1 in [(0, 4), (7, 10)]
    )
    furniture = []
    # Beds/tables are low 3D solids with rectangular floor footprints, not full-height walls.
    for i, name in enumerate(ROOMS):
        x = i * 6
        shift = rng.uniform(-0.35, 0.35)
        kind, appearance = (
            (
                "bed",
                "A low padded rectangular platform with a mattress, pillows and folded blanket.",
            )
            if name == "bedroom"
            else (
                "table",
                "A low broad horizontal tabletop, with four short legs and a smooth empty top.",
            )
        )
        furniture.append(
            (kind, Box((x + 2 + shift, 1.2, 0), (x + 4.2 + shift, 3.5, 0.75)), appearance)
        )
        furniture.append(
            (
                "cabinet",
                Box((x + 0.3, 8.8, 0), (x + 1.4, 9.6, 2.5)),
                "A tall wooden storage unit with solid closed doors; no visible shelves or books.",
            )
        )
        furniture.append(
            (
                "plant",
                Box((x + 4.8, 7.7, 0), (x + 5.4, 8.3, 1.7)),
                "A pot supporting green leaves and thin upright stems.",
            )
        )
    if not absent:
        x = list(ROOMS).index(target_room) * 6
        center = x + rng.uniform(2.6, 4.1)
        furniture.append(
            (
                "bookcase",
                Box((center - 0.7, 9.1, 0), (center + 0.7, 9.7, 2.7)),
                "A tall open wooden frame with several horizontal shelves holding rows of upright bound volumes with colored spines.",
            )
        )
    rng.shuffle(furniture)
    objects = tuple(
        Object(f"object_{i + 1}", kind, box, appearance)
        for i, (kind, box, appearance) in enumerate(furniture)
    )
    target_id = next((o.id for o in objects if o.kind == "bookcase"), None)
    # An unreachable sentinel disables Simulation's point-goal success. Search uses its own evaluator.
    world = Scenario(
        "unknown_furniture",
        seed,
        (2.0, 5.5, 1.4),
        (-100.0, -100.0, -100.0),
        walls + tuple(o.box for o in objects),
        (18.0, 10.0, 4.0),
    )
    return Home(world, objects, target_id)


def visible_objects(home, position, max_range=3.5, view_direction=None):
    """Ideal panoramic semantic/depth sensor: surface samples, range and solid occlusion.

    A visible sample unlocks a hand-authored appearance, not an RGB recognition result.
    No bounds, kind, hidden target ID, or unseen coordinates leave this sensor.
    """
    observations = []
    for obj in home.objects:
        axes = [[a, (a + b) / 2, b] for a, b in zip(obj.box.lo, obj.box.hi)]
        visible = []
        for point in itertools.product(*axes):
            delta = subtract(point, position)
            distance = norm(delta)
            if not 1e-6 < distance <= max_range:
                continue
            if (
                view_direction is not None
                and sum(a * b for a, b in zip(unit(delta), unit(view_direction))) < 0.5
            ):
                continue
            # Includes the object's own geometry: rear faces cannot be seen through it.
            if any(
                (hit := ray_box(position, delta, box)) is not None and hit < 1 - 1e-6
                for box in home.world.obstacles
            ):
                continue
            visible.append((distance, point))
        if visible:
            distance, point = min(visible)
            observations.append(
                {
                    "id": obj.id,
                    "appearance": obj.appearance,
                    "visible_surface_xyz": list(point),
                    "range_metres": round(distance, 2),
                    "room": room_at(point),
                }
            )
    return observations


def update_memory(memory, observations, position, now):
    for observation in observations:
        point = observation["visible_surface_xyz"]
        # A standoff point along the actually observed ray; no hidden geometry is consulted.
        direction = unit(subtract(position, point))
        approach = [p + d * 1.5 for p, d in zip(point, direction)]
        old = memory.get(observation["id"], {})
        memory[observation["id"]] = {
            **observation,
            "first_seen_at": old.get("first_seen_at", now),
            "last_seen_at": now,
            "approach_xyz": old.get("approach_xyz", approach),
        }


def search_waypoints():
    """Candidate viewpoints from room bounds only. No object-aware placement or ranking."""
    points = {}
    for room, data in ROOMS.items():
        x = data["bounds_xyz"][0][0]
        for side, dx, y in [
            ("south_west", 1.6, 2.5),
            ("south_east", 4.4, 2.5),
            ("north_west", 1.6, 7.5),
            ("north_east", 4.4, 7.5),
        ]:
            points[f"search_{room}_{side}"] = {
                "xyz": [x + dx, y, 1.4],
                "room": room,
                "description": f"Look around the {side} part of {room}.",
            }
        points[f"search_{room}_high"] = {
            "xyz": [x + 3, 6, 2.8],
            "room": room,
            "description": f"Look across {room} from higher altitude; useful for occluded low furniture.",
        }
    for i, portal in enumerate(PORTALS):
        for room, point in portal["approaches"].items():
            points[f"door_{i}_{room}"] = {
                "xyz": point,
                "room": room,
                "description": f"Approach doorway between {' and '.join(portal['rooms'])} on its {room} side.",
            }
    return points


def distance_to_box(position, box):
    return norm(tuple(max(a - p, 0, p - b) for p, a, b in zip(position, box.lo, box.hi)))


def evaluate_report(choice, home, sim, visible):
    """Evaluator only: model reports are never silently corrected."""
    if choice == "report_not_found":
        return "correct_absence" if home.target_id is None else "false_absence"
    object_id = choice.removeprefix("report_found_")
    obj = next((o for o in home.objects if o.id == object_id), None)
    if obj is None or object_id != home.target_id:
        return "wrong_object"
    if object_id not in {o["id"] for o in visible}:
        return "unverified_report"
    if distance_to_box(sim.position, obj.box) > 2.15 or norm(sim.velocity) > 0.15:
        return "premature_report"
    return "success"


def ground_truth(home):
    return {"objects": [asdict(o) for o in home.objects], "target_id": home.target_id}
