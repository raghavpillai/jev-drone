"""Audit the two demonstration flights and select a recording without hiding failures."""
import json
from pathlib import Path

import numpy as np

from jev_drone.audit.native import audit


ROOT = Path('results/px4-house-find-video')
OUT = Path('report/px4-find-video')


def summarize(path):
    row = audit(path)
    result = json.loads(path.read_text())
    journal = [json.loads(line) for line in path.with_name('calls.jsonl').read_text().splitlines()]
    objective = result['world']['mission'][0]
    rooms, entered, seen, seen_in_room = [], None, None, None
    for call, raw in zip(result['calls'], journal):
        observation = raw['state']['sensors']
        room = observation['room']
        if not rooms or rooms[-1] != room:
            rooms.append(room)
        visible = any(d['marker'] == objective['marker'] for d in observation['detections'])
        if seen is None and visible:
            seen = call['time']
        if room == objective['room']:
            if entered is None:
                entered = call['time']
            if seen_in_room is None and visible:
                seen_in_room = call['time']
    inspected = next((r['time'] for r in result['reports'] if r['stage'] == 0 and r['valid']), None)
    trace = result['trace']
    positions = np.array([t['position'] for t in trace])
    durations = np.diff([t['time'] for t in trace], prepend=trace[0]['time'])
    row.update(
        objective=objective, room_visits=rooms, entered_target_room=entered,
        first_seen=seen, first_seen_in_target_room=seen_in_room, inspected=inspected,
        false_reports=sum(not r['valid'] for r in result['reports']),
        api_errors=sum(bool(c.get('error')) for c in result['calls']),
        path_metres=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        stationary_seconds=float(sum(dt for t, dt in zip(trace, durations) if np.linalg.norm(t['velocity']) < .05)),
        selected_replans=sum(bool(c.get('accepted')) and c.get('choice') == 'brake_and_replan'
                             for c in result['calls'] if c['role'] == 'control'),
    )
    return row


def rank(row):
    return (not row['strict_pass'], row['inspected'] is None,
            row['contacts'], row['rule_violations'], row['false_reports'],
            row['inspected'] if row['inspected'] is not None else float('inf'), row['path_metres'])


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    paths = [ROOT/f"{trial['case']}-{trial['seed']}"/'result.json' for trial in protocol['trials']]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit('Awaiting completed trials: ' + ', '.join(missing))
    rows = [summarize(path) for path in paths]
    selected = min(rows, key=rank)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        'selection_rule': protocol['description'], 'trials': rows,
        'selected': selected['directory'],
        'strict_passes': sum(row['strict_pass'] for row in rows),
        'total_trials': len(rows), 'total_cost_usd': sum(row['cost_usd'] for row in rows),
        'scope': 'Two fresh find-only missions. Best-run demonstration selection, not a reliability estimate.',
    }
    (OUT/'comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Find-only demonstration missions', '', payload['scope'], '',
             '| Mission | Result | Verified find (sim s) | Contacts / rule violations | Path (m) |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        found = '—' if row['inspected'] is None else f"{row['inspected']:.1f}"
        lines.append(f"| {row['id']} | {'Strict pass' if row['strict_pass'] else row['status']} | {found} | "
                     f"{row['contacts']} / {row['rule_violations']} | {row['path_metres']:.1f} |")
    lines += ['', f"Selected: **{selected['id']}**. Both outcomes and independent audits are preserved in comparison.json.", '',
              'Selection prioritizes an audit-clean success, verified inspection, safety, then time and path length. '
              'The two seeds use established furniture recipes, not independently randomized houses. '
              'The unchanged v36 Jev planner/controller ran through OpenRouter; no control decisions were replaced.', '',
              f"Reported API cost: ${payload['total_cost_usd']:.6f}.", '']
    (OUT/'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
