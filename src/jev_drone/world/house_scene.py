"""Eight furnished rooms. This recipe is simulator/evaluator input, never Jev input."""
import numpy as np

from jev_drone.world.layout import HOUSE


def build(scene):
    box = scene.box
    for name, center, half in (
        ("floor", (12, 6, -.1), (12, 6, .1)), ("ceiling", (12, 6, 3.3), (12, 6, .1)),
        ("west_wall", (-.1, 6, 1.6), (.1, 6, 1.6)), ("east_wall", (24.1, 6, 1.6), (.1, 6, 1.6)),
        ("south_wall", (12, -.1, 1.6), (12, .1, 1.6)), ("north_wall", (12, 12.1, 1.6), (12, .1, 1.6))):
        box(name, center, half, (.78, .8, .82, 1))
    for coordinate in (6, 12, 18):
        scene._wall_with_door(0, coordinate, 0, 12, [3, 9], 1.65)
    scene._wall_with_door(1, 6, 0, 24, [3, 9, 15, 21], 1.65)
    from jev_drone.world.furnishings import furnish
    for room, (lo, hi) in HOUSE.rooms.items():
        furnish(scene, room, lo)
    case = scene.config.case
    if case in ("house_hidden", "house_sequence", "house_search", "house_search_absent"):
        box("workshop_search_screen", (21.35, 10.15, 1.15), (.10, .8, 1.15))
        box("library_low_partition", (16.2, 8.0, .65), (.12, .65, .65))
    if case == "house_blocked":
        box("unexpected_closed_door", (12, 3, 1.35), (.08, .825, 1.35), (.4, .28, .16, 1))
    scene.start = np.array([22.5, 9., 1.3] if case == "house_reverse" else [1.4, 3., 1.3])
    if case in ("house_search", "house_search_absent"):
        scene.start = np.array([19.1, 9., 1.3])
    jitter = scene.rng.uniform(-.2, .2, 2)
    positions = {"red": [22.65, 10.0+jitter[0], 1.6],
                 "blue": [1.4, 10.4+jitter[1], 1.6], "yellow": [22.6, 4.5, 1.6]}
    if case == "house_reverse":
        positions["red"] = [1.35, 4.2+jitter[0], 1.6]
    colors = {"red": (.95, .06, .06, 1), "blue": (.06, .12, .95, 1), "yellow": (.95, .9, .04, 1)}
    for name, point in positions.items():
        if name == "red" and case == "house_search_absent":
            continue
        box(name+"_target_stand", (point[0], point[1], .7), (.16, .16, .7), (.49, .45, .37, 1))
        body = box(name+"_marker", point, (.2, .2, .2), colors[name])
        scene.targets[name] = {"position": point, "body": body}
    scene.mission = [{"marker": "red", "room": "entry" if case == "house_reverse" else "workshop"}]
    if case == "house_sequence":
        scene.mission += [{"marker": "blue", "room": "study"}, {"marker": "yellow", "room": "kitchen"}]
    scene.mission.append({"dock": scene.start.tolist()})
