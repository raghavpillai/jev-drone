"""Export recorded Jev inputs, selections and applied controls for the video HUD."""
import argparse
import hashlib
import json
from pathlib import Path


def export(source, output):
    result = json.loads((source / 'result.json').read_text())
    journal = [json.loads(line) for line in (source / 'calls.jsonl').read_text().splitlines()]
    assert len(journal) == len(result['calls'])
    decisions = []
    for call, raw in zip(result['calls'], journal):
        assert call.get('answers') == raw.get('answers')
        state = raw['state']; sensors = state['sensors']
        selection = (raw.get('answers') or {}).get('selection', {})
        memory = state.get('search_memory', {})
        survey = state.get('local_survey', {})
        progress = state.get('mission_progress', state)
        choice = call.get('choice')
        option = state.get('options', {}).get(choice, {})
        decisions.append({
            'time': call.get('response_time', call['time']), 'request_time': call['time'],
            'role': call['role'], 'choice': choice, 'error': call.get('error'),
            'accepted': call.get('accepted'), 'rejection': call.get('rejection_reason'),
            'latency_ms': raw['latency_seconds'] * 1000,
            'probabilities': selection.get('probabilities', {}),
            'room': sensors['room'], 'position': sensors['position_estimate'],
            'objective': progress.get('objective'), 'task_kind': state.get('selected_task_kind'),
            'task': state.get('task'), 'option': option,
            'visible': progress.get('target_visible', False),
            'ready': progress.get('inspection_ready', False) or progress.get('dock_ready', False),
            'target_distance': progress.get('target_distance'),
            'clearance': sensors['body_clearance_m'], 'depth_status': state.get('body_depth_status', {}),
            'goal': state.get('goal'), 'offset': state.get('body_goal_offset_m'),
            'heading_error': state.get('goal_bearing_error_degrees'),
            'path_status': state.get('goal_path_status'), 'altitude_access': state.get('altitude_access'),
            'detour': state.get('active_detour'),
            'viewpoints_seen': sum(bool(v.get('inward_view_observed')) for v in memory.values()),
            'viewpoints_total': len(memory), 'heading_sectors': survey.get('surveyed_heading_sectors', []),
            'recent_tasks': [{k: task.get(k) for k in ('name', 'outcome')} for task in state.get('recent_tasks', [])[-2:]],
            'patch': (call.get('control_receipt') or {}).get('patch'),
        })
    epoch = result['world']['mission_start_sim_time']
    controls = [{'time': c['time'], 'revision': c['after']['revision'], 'values': c['after']['body_controls'],
                 'choice': c['choice'], 'patch': c['patch'],
                 'source': 'Jev update' if c['expected_revision'] is not None else 'Neutral reset'} for c in result['commands']]
    controls += [{'time': c['sim']-epoch, 'revision': c['after']['revision'], 'values': c['after']['body_controls'],
                  'choice': c['reason'], 'patch': None,
                  'source': 'Lease expired' if c['reason']=='lease_expired' else 'Telemetry stale'} for c in result['command_expirations']]
    controls.sort(key=lambda c: (c['time'], c['revision']))
    tag = source.parent.name + '--' + source.name
    payload = {'episode': 'data/' + tag + '.json', 'decisions': decisions, 'controls': controls,
               'provenance': {'source': str(source), 'model': next(c['model'] for c in result['calls'] if c.get('model')),
                              'request_journal_sha256': hashlib.sha256((source/'calls.jsonl').read_bytes()).hexdigest(),
                              'scope': 'Recorded request snapshots, returned choice scores and actual control-state events. No hidden model activations or reasoning are available.'}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(',', ':')))
    print(json.dumps({'output': str(output), 'decisions': len(decisions), 'control_events': len(controls)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.out)
