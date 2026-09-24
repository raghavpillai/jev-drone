"""Known semantic map supplied to the global planner; no route search."""
from experiments.legacy.sim import Box, Scenario

ROOMS = {
    "living_room": {"bounds_xyz": [[0, 0, 0], [7, 12, 4]]},
    "bedroom": {"bounds_xyz": [[7, 0, 0], [14, 12, 4]]},
}
PORTALS = [{"connects": ["living_room", "bedroom"],
            "opening_xyz": [[6.8, 4, 0], [7.2, 8, 4]],
            "approach_waypoints": ["door_living_side", "door_bedroom_side"]}]
WAYPOINTS = {
    "living_room_center": {"xyz": [3, 6, 1], "description": "Open center of living room."},
    "door_living_side": {"xyz": [5.5, 6, 1], "description": "Approach the connecting doorway from inside the living room."},
    "door_bedroom_side": {"xyz": [8.5, 6, 1], "description": "Exit the connecting doorway inside the bedroom."},
    "bedroom_center": {"xyz": [11, 6, 1], "description": "Open center of bedroom."},
}
DESTINATIONS = {
    "bedroom": {"room": "bedroom", "xyz": [11, 9.5, 1], "description": "Stop at the bedroom's destination marker."},
    "living_room_bookshelf": {"room": "living_room", "object_xyz": [2, 10.8, 1.4],
                              "xyz": [2, 9.2, 1], "description": "Stop in front of the bookshelf in the living room. xyz is its approach point, not inside the shelf."},
}


def locate_room(position):
    """Exact geometry-to-label conversion; does not choose a route."""
    for name, room in ROOMS.items():
        lo, hi = room["bounds_xyz"]
        if all(a <= p < b for p, a, b in zip(position, lo, hi)):
            return name
    return "outside_known_rooms"


def world(destination, start_room="living_room"):
    obstacles = (
        Box((6.8, 0, 0), (7.2, 4, 4)), Box((6.8, 8, 0), (7.2, 12, 4)),
        Box((1, 10.4, 0), (3, 11.2, 2.8)),
        Box((3.6, 3.8, 0), (4.4, 4.6, 4)),
        Box((9.2, 7.1, 0), (10.2, 8.1, 1.8)),
    )
    start = (2., 2., 1.) if start_room == "living_room" else (11., 9.5, 1.)
    return Scenario("house", 0, start, tuple(DESTINATIONS[destination]["xyz"]), obstacles, (14., 12., 4.))
