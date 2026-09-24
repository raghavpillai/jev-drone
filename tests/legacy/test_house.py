from experiments.legacy.house import DESTINATIONS, WAYPOINTS, locate_room, world
from experiments.legacy.sim import Simulation, norm, ray_box, subtract


def test_room_labels_use_geometry():
    assert locate_room((2, 2, 1)) == "living_room"
    assert locate_room((11, 9.5, 1)) == "bedroom"
    assert locate_room((15, 2, 1)) == "outside_known_rooms"


def test_object_destination_is_an_approach_point_outside_solid_geometry():
    for name in DESTINATIONS:
        room = world(name)
        for box in room.obstacles:
            assert ray_box(room.goal, (0, 0, 0), box.expanded(0.2)) is None


def test_waypoint_observation_does_not_change_mission_completion_target():
    sim = Simulation(world("bedroom"))
    final = sim.world.goal
    target = tuple(WAYPOINTS["door_living_side"]["xyz"])
    observation = sim.observation([], target=target)
    assert observation["goal"]["distance_metres"] == round(norm(subtract(target, sim.position)), 2)
    assert sim.world.goal == final
    sim.position = target
    sim.advance(0.6)
    assert sim.status == "running"


def test_authored_portal_segment_is_clear_for_vehicle():
    room = world("bedroom")
    start = tuple(WAYPOINTS["door_living_side"]["xyz"])
    end = tuple(WAYPOINTS["door_bedroom_side"]["xyz"])
    for box in room.obstacles:
        hit = ray_box(start, subtract(end, start), box.expanded(0.2))
        assert hit is None or hit > 1
