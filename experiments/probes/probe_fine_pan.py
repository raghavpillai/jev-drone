"""Compare a finer pan choice on recorded doorway-alignment requests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.control.controls import changed_channels


OLD = 'For a requested turn, brake and pan in place. Under about 25 degrees use slow\npan to avoid overshooting; under 8 degrees stop pan.'
NEW = 'For a requested turn, brake and pan in place. Under about 25 degrees use\npan_left_creep/pan_right_creep (0.08 rad/s) for final alignment; slow pan is 0.2\nrad/s and full pan is 0.6 rad/s. Under 8 degrees stop pan. Let measured yaw settle\nbefore correcting again; a released pan takes time to stop.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases, negatives = [], {}
    for trial in ['house_hidden-5311', 'house_reverse-5312']:
        path = Path('results/px4-house-v36-validation') / trial / 'calls.jsonl'
        selected = set()
        for index, line in enumerate(path.read_text().splitlines()):
            call = json.loads(line)
            state = call['state']
            if 'doorway_phase' not in state:
                continue
            error = state['doorway_alignment']['heading_error_degrees']
            group = (state['task']['name'], error > 0)
            case = {'label': f'{trial}-{index}', 'source': str(path), 'index': index, 'call': call}
            if 8 < abs(error) < 25 and group not in selected and len(selected) < 3:
                cases.append({**case, 'positive': True})
                selected.add(group)
            negative = 'aligned' if abs(error) < 8 else 'far_heading' if abs(error) > 60 else None
            if negative and negative not in negatives:
                negatives[negative] = {**case, 'positive': False}
    cases.extend(negatives.values())
    (args.out / 'cases.json').write_text(json.dumps(cases) + '\n')
    (args.out / 'protocol.json').write_text(json.dumps({'variants': ['original', 'fine_pan'], 'repeats': 2,
        'cases': [{k: v for k, v in c.items() if k != 'call'} for c in cases],
        'scope': 'Recorded-state development choices from frozen v36 validation. Tests Jev use of new pan controls, '
                 'not physical stopping or mission completion. Native physics must verify overshoot improvement.'}, indent=2) + '\n')
    gateway = JevGateway('openrouter', load_credential('openrouter', Path('.env')),
                         journal=args.out / 'calls.jsonl')
    rows = []
    try:
        for case in cases:
            for repeat in range(2):
                for variant in (['original', 'fine_pan'] if repeat == 0 else ['fine_pan', 'original']):
                    state = deepcopy(case['call']['state'])
                    question = deepcopy(case['call']['questions']['selection'])
                    if variant == 'fine_pan':
                        assert OLD in question['instructions']
                        question['instructions'] = question['instructions'].replace(OLD, NEW)
                        for side, rate in [('left', .08), ('right', -.08)]:
                            name = f'pan_{side}_creep'
                            old_name = f'pan_{side}_slow'
                            effect = deepcopy(state['control_effects'][old_name])
                            effect['resulting_controls']['yaw_rate_rps'] = rate
                            effect['patch'] = changed_channels(state['current_controls']['body_controls'], effect['resulting_controls'])
                            state['control_effects'][name] = effect
                            question['criteria'][name] = ('Apply ' + json.dumps(effect['patch']) + '; retain omitted channels. '
                                'Fine pan for final alignment, 0.08 rad/s. Turn effect: ' + effect['turn_effect']
                                + '. Movement violations: ' + json.dumps(effect['movement_violations']))
                    result = gateway.choose(state, question['instructions'], question['criteria'])
                    choice = result.get('choice')
                    effect = state['control_effects'].get(choice, {})
                    row = {'case': case['label'], 'positive': case['positive'], 'variant': variant,
                           'repeat': repeat, 'choice': choice, 'error': result.get('error'),
                           'fine_pan_toward_heading': choice in ('pan_left_creep', 'pan_right_creep')
                               and effect.get('turn_effect') == 'turn_toward_heading',
                           'violations': effect.get('movement_violations')}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
