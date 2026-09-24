"""Describe room boundaries during return without selecting a route or control."""
from copy import deepcopy


INSTRUCTIONS = """
For return_to_launch, first return to the launch ROOM, then reach the launch
position and report dock when ready. The outbound pose history includes search
sweeps, failed approaches and loops; it is NOT a prescribed return route. A
successful retrace step is not a reason to replay every remaining search detour.
Outside the launch room, choose change_room toward launch using the known room
connections and doorway tasks. Use retrace or transit as local help when an
approach is blocked, then reconsider the room transition. Observed obstructions
still matter: this does not declare any door or straight segment clear.
A retrace coordinate in another room requires passing its doorway. Prefer the
actual approach/cross task for that opening so the pilot gets its centerline,
crossing altitude and heading. A remembered point across a wall is not a bypass.
Within the launch room, return_launch is useful when its approach is observed
clear; otherwise choose an intermediate observed route around the obstruction.
Do not restart room search on return. All task and control choices remain yours.
"""


def describe(state, criteria):
    if 'dock' not in state.get('objective', {}):
        return state, criteria
    state, criteria = deepcopy(state), dict(criteria)
    current = state['sensors']['room']
    launch = state['ACTIVE_OBJECTIVE']['room']
    state['return_progress'] = {
        'phase': 'DOCK_WITHIN_LAUNCH_ROOM' if current == launch else 'RETURN_TO_LAUNCH_ROOM',
        'current_room': current, 'launch_room': launch,
        'outbound_pose_history_includes_search_detours': True,
    }
    for name, option in state['options'].items():
        if not name.startswith('retrace_'):
            continue
        destination = option.get('room')
        if destination is None or destination == current:
            continue
        door = next((edge['door'] for edge in state['known_room_connections'].get(current, [])
                     if edge['room'] == destination), None)
        option['requires_room_transition'] = True
        option['connecting_door'] = door
        option['available_doorway_tasks'] = [prefix+door for prefix in ('approach_', 'cross_')
                                            if door and prefix+door in state['options']]
        if name in criteria:
            criteria[name] += (' This point is in another room. Cross the known opening with its doorway task; '
                               'do not treat a diagonal across the wall as a local bypass.')
    return state, criteria
