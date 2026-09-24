import numpy as np

from jev_drone.planning.door_memory import DoorMemory


def surface(axis):
    # Door centers: living/bedroom [6,2], bedroom/study [9,4].
    if axis == 0:
        return np.array(
            [
                [5.92, 2.0 + a, z]
                for a in np.linspace(-0.45, 0.45, 10)
                for z in np.linspace(0.85, 2.0, 10)
            ]
        )
    return np.array(
        [
            [9.0 + a, 3.92, z]
            for a in np.linspace(-0.45, 0.45, 10)
            for z in np.linspace(0.85, 2.0, 10)
        ]
    )


def test_only_observed_surfaces_mark_the_relevant_opening():
    memory = DoorMemory()
    memory.observe(np.empty((0, 3)), 0.0)
    assert not memory.obstructions
    memory.observe(surface(0), 2.0)
    assert set(memory.obstructions) == {"living_bedroom"}
    memory.observe(surface(1), 3.0)
    assert set(memory.obstructions) == {"living_bedroom", "bedroom_study"}


def test_jamb_and_lintel_points_do_not_claim_the_opening_is_blocked():
    memory = DoorMemory()
    cloud = surface(0)
    cloud[:, 1] = 3.0
    memory.observe(cloud, 1.0)
    cloud = surface(0)
    cloud[:, 2] = 2.5
    memory.observe(cloud, 2.0)
    assert not memory.obstructions


def test_obstruction_facts_do_not_remove_choices_or_expose_unseen_geometry():
    memory = DoorMemory()
    memory.observe(surface(0), 2.0)
    criteria = dict(
        approach_living_bedroom="Approach",
        cross_living_bedroom="Cross",
        approach_living_study="Alternate",
    )
    state, described = memory.describe({}, criteria)
    assert set(described) == set(criteria)
    assert "OBSTRUCTION" in described["cross_living_bedroom"]
    assert described["approach_living_study"] == "Alternate"
    assert set(state["observed_door_obstructions"]) == {"living_bedroom"}
