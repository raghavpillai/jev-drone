"""Offline candidate only; the paired probe did not justify enabling it in flight."""

INSTRUCTIONS = """
aligned_crossing_waiting_for_movement means the doorway heading, centerline and
altitude are aligned, the drone is already stopped, and a permitted forward
movement toward the unfinished crossing exists. A release whose resulting
controls are still all zero cannot advance this crossing. Choose a permitted
progressing control, then reassess depth. This never applies during braking,
arrival, objective reporting, non-door tasks, or absent safe observed clearance.
"""


def describe(state, criteria):
    if state.get('doorway_phase',{}).get('phase') != 'CROSS_OPENING':
        return state, criteria
    obs, progress = state['sensors'], state.get('mission_progress',{})
    if (state['at_position'] or state.get('door_passage_complete')
            or progress.get('target_in_inspection_range') or progress.get('dock_in_range')
            or any(state['current_controls']['body_controls'].values())
            or obs['speed'] > .1 or abs(obs['yaw_rate_rps']) > .1):
        return state, criteria
    effects = state['control_effects']
    if not any(not e['movement_violations'] and e.get('travel_effect',{}).get('forward')=='toward_goal'
               for e in effects.values()):
        return state, criteria
    state = {**state,'aligned_crossing_waiting_for_movement':True,
             'control_effects':{name:dict(e) for name,e in effects.items()}}
    criteria = dict(criteria)
    for name,effect in state['control_effects'].items():
        if (name in ('keep_controls','brake') or name.startswith('stop_')) and not any(effect['resulting_controls'].values()):
            effect['position_progress']='already_stopped_no_crossing_progress'
            criteria[name] += ' ALREADY STOPPED AND ALIGNED: this repeats a satisfied release and makes no crossing progress. An allowed forward movement exists.'
    return state,criteria
