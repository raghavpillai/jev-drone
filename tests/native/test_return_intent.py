from copy import deepcopy

from jev_drone.planning.planner_modes import MODES, available_modes
from jev_drone.planning.return_intent import describe
from jev_drone.world.layout import HOUSE


def state(room="library"):
    return {
        "objective": {"dock": [1.4, 3.0, 1.3]},
        "ACTIVE_OBJECTIVE": {"room": "entry"},
        "sensors": {"room": room},
        "known_room_connections": HOUSE.neighbors(),
        "options": {
            "retrace_8": {"position": [11.3, 8.8, 1.3], "room": "bedroom"},
            "retrace_7": {"position": [12.8, 9.0, 1.3], "room": "library"},
            "approach_bedroom_library": {"position": [13.0, 9.0, 1.3]},
            "cross_bedroom_library": {"position": [11.0, 9.0, 1.3]},
            "give_up": {},
        },
    }


def test_return_facts_expose_boundary_without_selecting_or_changing_waypoints():
    original = state()
    before = deepcopy(original)
    criteria = dict.fromkeys(original["options"], "Original.")
    updated, choices = describe(original, criteria)
    assert original == before and set(choices) == set(criteria)
    assert updated["return_progress"]["phase"] == "RETURN_TO_LAUNCH_ROOM"
    waypoint = updated["options"]["retrace_8"]
    assert waypoint["position"] == original["options"]["retrace_8"]["position"]
    assert waypoint["connecting_door"] == "bedroom_library"
    assert waypoint["available_doorway_tasks"] == [
        "approach_bedroom_library",
        "cross_bedroom_library",
    ]
    assert "requires_room_transition" not in updated["options"]["retrace_7"]


def test_no_crossing_is_fabricated_for_nonadjacent_room_and_docking_phase_is_local():
    original = state("entry")
    updated, _ = describe(original, dict.fromkeys(original["options"], ""))
    assert updated["return_progress"]["phase"] == "DOCK_WITHIN_LAUNCH_ROOM"
    assert updated["options"]["retrace_8"]["connecting_door"] is None
    assert updated["options"]["retrace_8"]["available_doorway_tasks"] == []


def test_outbound_search_remains_unchanged():
    original = {**state(), "objective": {"marker": "red", "room": "workshop"}}
    criteria = dict.fromkeys(original["options"], "Original.")
    updated, choices = describe(original, criteria)
    assert updated is original and choices is criteria


def test_return_has_no_marker_search_or_approach_but_retains_navigation_choices():
    assert set(available_modes(state())) == {
        "report",
        "change_room",
        "transit",
        "retrace",
        "give_up",
    }
    assert available_modes({"objective": {"marker": "red", "room": "workshop"}}) == MODES
