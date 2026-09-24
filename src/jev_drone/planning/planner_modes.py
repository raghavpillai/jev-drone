"""Jev selects the task family before selecting a concrete navigation task."""
from jev_drone.planning.return_intent import INSTRUCTIONS as RETURN_INSTRUCTIONS

INSTRUCTIONS = "You are Jev, planning a drone mission. Select the kind of task needed NOW. Read ACTIVE_OBJECTIVE, current room, readiness, task outcomes and options. Rooms and doors are known; furniture is observed only through cameras. This decision does not move the drone or complete an objective."

INSTRUCTIONS += RETURN_INSTRUCTIONS

MODES = {
    "report": "The current objective is observed ready (inspection_ready or dock_ready). Report it now.",
    "change_room": "Outside the ACTIVE_OBJECTIVE room: approach/cross a door toward it, directly or via another room. An available crossing after reaching a doorway is useful progress, not more searching.",
    "search": "In the requested room, requested marker not yet observed: inspect different viewpoints and headings.",
    "approach": "In the requested room with a known marker: approach it or use an intermediate viewpoint around an obstruction, then verify it.",
    "transit": "Inside the destination room returning to launch, or blocked far from an available door crossing: select an intermediate transit point. Do not oscillate between altitudes when a crossing toward the objective room is available.",
    "retrace": "Returning and local approach is blocked: consider a nearby observed outbound position as a local bypass; do not replay the entire search history.",
    "give_up": "No useful task remains feasible with available sensing/time. Unvisited viewpoints or an available crossing are reasons to continue, not give up.",
}


def available_modes(state):
    # A dock objective has no object to search for or approach. Navigation
    # alternatives remain Jev choices, including deliberate backtracking.
    if 'dock' in state.get('objective', {}):
        return {name: description for name, description in MODES.items() if name not in ('search', 'approach')}
    return dict(MODES)


def eligible(name, mode):
    if name == "give_up":
        return True
    doorway = name.startswith(("approach_", "cross_")) and not name.startswith("approach_marker")
    if mode == "report":
        return name in ("inspect", "dock")
    if mode == "change_room":
        return doorway
    if mode == "search":
        return name.startswith(("view_", "look_"))
    if mode == "approach":
        return name.startswith(("approach_marker", "view_", "look_"))
    if mode == "transit":
        return name == "return_launch" or name.startswith("transit_") or doorway
    if mode == "retrace":
        return name == "return_launch" or name.startswith("retrace_")
    return False
