"""Expose the room leg Jev previously selected; do not choose a route."""

INSTRUCTIONS = """
room_transit_intent is YOUR most recently selected doorway and next room.
Finish that room leg before reconsidering the final destination. In particular,
after approaching a door to an intermediate room, CROSS that same door when its
cross task is available; do not abandon the route merely because the intermediate
room is not the final objective room. If the CROSS task stalled against an
obstruction, reconsider the route and select another doorway. A stalled approach
near a doorway is different: an available crossing may already be the next step.
The remembered intention is not an automatic route or a safety guarantee.
"""


def describe(state, criteria):
    state, criteria = dict(state), dict(criteria)
    previous = next(
        (
            task
            for task in reversed(state.get("recent_tasks", []))
            if task.get("leads_to_room") is not None
            and task.get("mission_stage", 0) == state.get("stage", 0)
        ),
        None,
    )
    state["room_transit_intent"] = None
    if previous is None or previous["leads_to_room"] == state["sensors"]["room"]:
        return state, criteria
    door = previous["name"].split("_", 1)[1]
    state["room_transit_intent"] = {
        "door": door,
        "next_room": previous["leads_to_room"],
        "selected_task": previous["name"],
        "outcome": previous["outcome"],
        "crossing_available": "cross_" + door in state["options"],
    }
    for name in ("approach_" + door, "cross_" + door):
        if name not in criteria:
            continue
        criteria[name] += (
            " Continues the room leg you selected toward " + previous["leads_to_room"] + "."
        )
        if name.startswith("cross_"):
            criteria[name] += (
                " Crossing is available NOW; complete this leg before choosing the following room."
            )
    return state, criteria
