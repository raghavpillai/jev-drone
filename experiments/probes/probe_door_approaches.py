"""Offline-only Jev experiment: alternate approaches from known door geometry."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from jev_drone.gateway import JevGateway, load_credential
from experiments.probes.check_door_memory import recorded_cloud
from jev_drone.world.layout import HOUSE


def near_options(state, cloud):
    options = {}
    for door, center in HOUSE.doors.items():
        room = state['sensors']['room']
        if room not in HOUSE.connections[door]:
            continue
        destination = next(r for r in HOUSE.connections[door] if r != room)
        position = np.array(center)
        position[HOUSE.door_axis(door)] -= .65 * HOUSE.door_direction(door, destination)
        gap = round(float(np.linalg.norm(cloud - position, axis=1).min()) - .4, 3)
        delta = position - state['sensors']['position_estimate']
        options['approach_' + door + '_near'] = {
            'position': position.tolist(), 'door': door, 'leads_to_room': destination,
            'purpose': 'Alternative point 0.65 m before the known opening; Jev must fly and reassess it.',
            'observed_goal_gap_m': gap, 'observed_endpoint_unsafe': gap < .08,
            'observed_path_clearance_m': None,
            'distance_m': round(float(np.linalg.norm(delta)), 2),
            'already_at_position': bool(np.linalg.norm(delta[:2]) <= .35 and abs(delta[2]) <= .15)}
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for cohort, trial in [('px4-house-v35-return', 'house_hidden-5111'),
                         ('px4-house-v36-validation', 'house_hidden-5311'),
                         ('px4-house-v36-validation', 'house_reverse-5312')]:
        directory = Path('results') / cohort / trial
        result = json.loads((directory / 'result.json').read_text())
        journal = [json.loads(line) for line in (directory / 'calls.jsonl').read_text().splitlines()]
        frames = sorted((directory / 'result-frames').glob('*.npz'))
        selected = set()
        for index, (call, raw) in enumerate(zip(result['calls'], journal)):
            state = raw['state']
            if state.get('selected_task_kind') != 'change_room':
                continue
            bad = [name for name, option in state['options'].items()
                   if name.startswith('approach_') and name[9:] in HOUSE.doors and option.get('observed_endpoint_unsafe')]
            if not bad or (state['sensors']['room'], state['stage']) in selected:
                continue
            frame = max((f for f in frames if float(f.stem) <= call['time']), key=lambda f: float(f.stem))
            cloud = recorded_cloud(frame, result['world']['hfov'])
            alternatives = near_options(state, cloud)
            useful = [name + '_near' for name in bad if not alternatives[name + '_near']['observed_endpoint_unsafe']]
            if not useful:
                continue
            base = deepcopy(raw)
            # Both arms use the same archived cloud for endpoint evidence.
            for name, option in base['state']['options'].items():
                if 'position' not in option:
                    continue
                gap = round(float(np.linalg.norm(cloud - option['position'], axis=1).min()) - .4, 3)
                option.update(observed_goal_gap_m=gap, observed_endpoint_unsafe=gap < .08)
                if name in base['questions']['selection']['criteria']:
                    base['questions']['selection']['criteria'][name] = ('DO NOT SELECT this observed unsafe endpoint. ' if gap < .08 else '') + json.dumps(option)
            base['state']['offline_endpoint_depth_frame'] = {'path': str(frame), 'time': float(frame.stem),
                'lag_to_request_seconds': round(call['time'] - float(frame.stem), 3),
                'scope': 'Endpoint evidence only. Other recorded observations remain unchanged; this is not a flight.'}
            issue_remains = any(base['state']['options'][name.removesuffix('_near')]['observed_endpoint_unsafe']
                                for name in useful)
            cases.append({'label': f'{trial}-{index}', 'source': str(directory), 'index': index,
                          'probe_endpoint_status': 'unsafe_base_with_clear_alternative' if issue_remains else 'boundary_control_base_now_clear',
                          'useful_alternatives': useful, 'alternatives': alternatives, 'call': base})
            selected.add((state['sensors']['room'], state['stage']))
            if len(selected) == 2:
                break
    (args.out / 'cases.json').write_text(json.dumps(cases) + '\n')
    (args.out / 'protocol.json').write_text(json.dumps({'variants': ['single_approach', 'alternate_approaches'],
        'repeats': 2, 'cases': [{k: v for k, v in c.items() if k not in ('call', 'alternatives')} for c in cases],
        'scope': 'Development probe only, not enabled in native policy. All adjacent doors receive the same '
                 'architecturally generated near candidate. Endpoint evidence comes from archived depth, never '
                 'evaluator furniture. Same endpoint cloud in both arms; frame may precede original request. '
                 'Positive gap is not route certification. No claims of physical completion.'}, indent=2) + '\n')
    gateway = JevGateway('openrouter', load_credential('openrouter', Path('.env')),
                         journal=args.out / 'calls.jsonl')
    rows = []
    try:
        for case in cases:
            for repeat in range(2):
                for variant in (['single_approach', 'alternate_approaches'] if repeat == 0 else ['alternate_approaches', 'single_approach']):
                    state = deepcopy(case['call']['state'])
                    question = deepcopy(case['call']['questions']['selection'])
                    if variant == 'alternate_approaches':
                        state['options'].update(case['alternatives'])
                        for name, option in case['alternatives'].items():
                            question['criteria'][name] = ('DO NOT SELECT this observed unsafe endpoint. ' if option['observed_endpoint_unsafe'] else '') + json.dumps(option)
                        question['instructions'] += '\nAn unsafe approach point does not prove the whole doorway blocked. Alternative near approaches are existing-room positions before the same opening; consider their observed endpoint evidence. Unknown path clearance still needs local observation and Jev control. No route is automatically flown.'
                    response = gateway.choose(state, question['instructions'], question['criteria'])
                    choice = response.get('choice')
                    row = {'case': case['label'], 'repeat': repeat, 'variant': variant, 'choice': choice,
                           'selected_near_candidate': choice in case['useful_alternatives'],
                           'probe_endpoint_status': case['probe_endpoint_status'],
                           'selected_observed_unsafe_endpoint': state['options'].get(choice, {}).get('observed_endpoint_unsafe', False),
                           'error': response.get('error')}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
