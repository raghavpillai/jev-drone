"""Evaluate recorded doorway time and turning; never used by the flight policy."""
from collections import Counter


def analyze(result, journal):
    controls = [(call, raw['state']) for call, raw in zip(result['calls'], journal)
                if call['role'] == 'control']
    rows = []
    for task in result['tasks']:
        if not task['name'].startswith('cross_'):
            continue
        end = task.get('ended', result['sim_seconds'])
        samples = [(call, state) for call, state in controls
                   if task['time'] <= call['time'] < end and state['task']['name'] == task['name']]
        pans = []
        for call, state in samples:
            if not call.get('accepted'):
                continue
            rate = state['control_effects'][call['choice']]['resulting_controls']['yaw_rate_rps']
            if rate:
                pans.append(1 if rate > 0 else -1)
        rows.append({'task': task['name'], 'mission_stage': task.get('mission_stage', 0),
                     'start': task['time'], 'end': end, 'duration': end - task['time'],
                     'outcome': task['outcome'], 'controls': len(samples),
                     'pan_direction_changes': sum(a != b for a, b in zip(pans, pans[1:])),
                     'phase_samples': dict(Counter(s.get('doorway_phase', {}).get('phase', 'other')
                                                   for _, s in samples))})
    return {'scope': 'Recorded task durations and request samples. Pan sign changes include legitimate '
                     'corrections and are not automatically failures. Samples are not time-weighted.',
            'attempts': len(rows), 'outcomes': dict(Counter(r['outcome'] for r in rows)),
            'total_seconds': sum(r['duration'] for r in rows),
            'pan_direction_changes': sum(r['pan_direction_changes'] for r in rows),
            'tasks': rows}
