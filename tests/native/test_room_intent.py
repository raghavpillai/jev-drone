from jev_drone.planning.room_intent import describe


def test_intermediate_room_intention_survives_a_stalled_approach():
    state = {
        "objective": {"room": "bedroom"},
        "sensors": {"room": "living"},
        "recent_tasks": [
            {"name": "approach_living_study", "leads_to_room": "study", "outcome": "stalled"}
        ],
        "options": {"cross_living_study": {}, "approach_living_bedroom": {}, "give_up": {}},
    }
    criteria = dict.fromkeys(state["options"], "Original.")
    enriched, described = describe(state, criteria)
    assert enriched["room_transit_intent"]["next_room"] == "study"
    assert enriched["room_transit_intent"]["crossing_available"]
    assert "available NOW" in described["cross_living_study"]
    assert set(described) == set(criteria)
    assert "room_transit_intent" not in state
    entered, _ = describe({**state, "sensors": {"room": "study"}}, criteria)
    assert entered["room_transit_intent"] is None
