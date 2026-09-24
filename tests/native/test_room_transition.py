from copy import deepcopy

from jev_drone.planning.room_transition import describe
from jev_drone.world.layout import HOUSE


def example():
    return {
        "stage": 0,
        "sensors": {"room": "living"},
        "recent_tasks": [
            {
                "name": "cross_entry_living",
                "leads_to_room": "living",
                "outcome": "arrived",
                "mission_stage": 0,
            }
        ],
        "options": {
            "cross_entry_living": {"leads_to_room": "entry"},
            "approach_living_dining": {"leads_to_room": "dining"},
        },
    }


def test_same_named_door_now_goes_back_and_remains_selectable():
    state = example()
    before = deepcopy(state)
    criteria = dict.fromkeys(state["options"], "Original.")
    enriched, described = describe(state, criteria, HOUSE)
    assert enriched["last_completed_room_transition"]["from_room"] == "entry"
    assert enriched["last_completed_room_transition"]["to_room"] == "living"
    assert enriched["options"]["cross_entry_living"]["reverses_completed_crossing"]
    assert "reverses_completed_crossing" not in enriched["options"]["approach_living_dining"]
    assert set(described) == set(criteria) and "BACK to entry" in described["cross_entry_living"]
    assert state == before


def test_new_mission_stage_does_not_discourage_legitimate_return():
    state = example()
    state["stage"] = 1
    assert describe(state, {}, HOUSE) == (state, {})


def test_failed_crossing_or_already_left_destination_does_not_claim_completion():
    state = example()
    state["recent_tasks"][0]["outcome"] = "stalled"
    assert describe(state, {}, HOUSE) == (state, {})
    state = example()
    state["sensors"]["room"] = "entry"
    assert describe(state, {}, HOUSE) == (state, {})
