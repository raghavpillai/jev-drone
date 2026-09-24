"""Describe progress and bypass visibility without choosing or filtering actions."""
def enrich(state, criteria):
    state = dict(state)
    state['control_effects'] = {name: dict(effect) for name, effect in state['control_effects'].items()}
    criteria = dict(criteria)
    observation = state['sensors']
    limits = state.get('control_limits', {})
    slow, full = limits.get('slow_clearance_m', .4), limits.get('full_clearance_m', 1.3)
    creep = limits.get('creep_clearance_m', slow)
    current = state['current_controls']['body_controls']
    state['holding_position_away_from_goal'] = not any(current.values()) and not state['at_position']
    path_clearance = state.get('goal_path_clearance_m')
    state['goal_path_status'] = ('ARRIVED' if state['at_position'] else
                                'UNKNOWN' if path_clearance is None else
                                'BLOCKED' if path_clearance < creep else
                                'CREEP_ONLY' if path_clearance < slow else 'OBSERVED_CLEAR')
    state['body_depth_status'] = {name: 'UNKNOWN' if value is None else 'OPEN' if value >= full else
                                'SLOW_ONLY' if value >= slow else 'CREEP_ONLY' if value >= creep else 'BLOCKED'
                                for name, value in observation['body_clearance_m'].items()}
    for name, effect in state['control_effects'].items():
        if name.startswith('detour_'):
            prefix, distance = next((p, d) for p, d in
                (("detour_short_", .6), ("detour_medium_", .9), ("detour_", 1.3)) if name.startswith(p))
            direction = name[len(prefix):]
            depth = observation['body_clearance_m'][direction]
            effect['bypass_direction'] = direction
            effect['bypass_depth_m'] = depth
            effect['bypass_visibility'] = state['body_depth_status'][direction]
            effect['bypass_distance_m'] = distance
            effect['bypass_viable'] = depth is not None and depth >= max(distance, creep)
            effect['position_goal_reached'] = state['at_position']
            criteria[name] += f' Observed {direction} depth: {depth}m; {effect["bypass_visibility"]}. Requested step: {distance}m. '
            criteria[name] += ('This selects a reachable temporary bypass destination around an obstruction; YOU fly toward it on following calls.'
                               if effect['bypass_viable'] else
                               'Insufficient observed space for this detour. Do not select this bypass.')
            if state['at_position']:
                criteria[name] += ' Position goal is ALREADY REACHED: do not leave it on a detour. Finish the requested turn in place.'
        if name == 'keep_controls' and not any(current.values()):
            effect['progress'] = 'continue_hovering'
            criteria[name] += ' All current controls are ZERO. This stays in place and makes NO navigation progress.'
    return state, criteria


INSTRUCTIONS = '''
FIRST check at_position. If true, stop translation and complete the requested
heading IN PLACE, then stop pan. Do not choose a detour: there is no remaining
position to reach. A wall in front does not stop rotation in place.
When goal_path_status is BLOCKED, not at_position, and you are stopped, do NOT keep_controls forever:
that only keeps hovering. Select a detour with bypass_viable=true. Normal detours
are 1.3m, detour_medium_DIRECTION is 0.9m, and detour_short_DIRECTION is 0.6m.
Use a smaller step when a full step does not fit, for example ascending above
furniture when the ceiling limits available space. Prefer a route whose second
leg also clears observed surfaces; the smallest step need not clear the obstacle.
This changes the temporary local goal so that your following control decisions can
fly around/above the obstacle. A bypass may intentionally move away from the original
goal; that is necessary to get around furniture. A detour must fit entirely within
the observed directional clearance. If no bypass is viable,
pan to acquire another view or wait briefly for a moving obstacle. Never repeat
hovering indefinitely when a viable bypass exists. After the bypass, reassess the
original goal; another bypass step may be needed. All steering remains your choice.
If goal_path_status is OBSERVED_CLEAR, travel toward it with the appropriate axes;
if CREEP_ONLY, use an allowed creep intent for the narrow observed space.
do not invent an unnecessary bypass. UNKNOWN calls for looking toward the goal
to get depth, not for repeatedly moving away. Use recent_detours to avoid repeating
a bypass that returned you to the same blocked approach. Prefer lateral or vertical
bypasses past furniture; backing up helps gain space but alone does not pass it.
'''
