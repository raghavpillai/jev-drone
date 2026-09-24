"""Make completed room transitions explicit without selecting the next doorway."""

INSTRUCTIONS = """
last_completed_room_transition records a doorway crossing you already finished
for this mission stage. You are now in its destination room. Selecting the same
cross_DOOR name here reverses direction and returns to the previous room; it
does NOT continue the completed crossing. Prefer a useful onward room connection
toward ACTIVE_OBJECTIVE. Backtracking remains available if the objective is in
the previous room or a deliberate alternate route requires it after obstruction.
Do not bounce across a door just because its crossing option is available.
"""


def describe(state, criteria, layout):
    previous = next(
        (
            t
            for t in reversed(state.get("recent_tasks", []))
            if t["name"].startswith("cross_")
            and t.get("outcome") == "arrived"
            and t.get("mission_stage", 0) == state["stage"]
        ),
        None,
    )
    room = state["sensors"]["room"]
    if previous is None or previous.get("leads_to_room") != room:
        return state, criteria
    door = previous["name"][6:]
    origin = next(r for r in layout.connections[door] if r != room)
    state, criteria = dict(state), dict(criteria)
    state["last_completed_room_transition"] = {
        "door": door,
        "from_room": origin,
        "to_room": room,
        "mission_stage": state["stage"],
        "outcome": "already completed",
    }
    state["options"] = {name: dict(option) for name, option in state["options"].items()}
    for name, option in state["options"].items():
        if option.get("leads_to_room") != origin:
            continue
        option["reverses_completed_crossing"] = True
        if name in criteria:
            criteria[name] += (
                " This goes BACK to "
                + origin
                + " across the door you already crossed into "
                + room
                + ". It is not a continuation of that completed crossing. Use only for deliberate backtracking."
            )
    return state, criteria
