"""Check altitude wording away from the original near-goal failure cases."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from jev_drone.gateway import JevGateway, load_credential
from experiments.representations.altitude_access import describe, INSTRUCTIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--uniform',action='store_true');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);cases=[];counts={}
    for trial in ('house_hidden-5311','house_reverse-5312'):
        path=Path('results/px4-house-v36-validation')/trial/'calls.jsonl'
        for index,line in enumerate(path.read_text().splitlines()):
            c=json.loads(line);s=c['state']
            if 'control_effects' not in s:continue
            updated,_=describe(s,c['questions']['selection']['criteria']);fact=updated.get('altitude_access',{})
            names=('up','up_creep') if s['body_goal_offset_m']['up']>0 else ('down','down_creep')
            reasons=[reason for name in names for reason in s['control_effects'][name]['movement_violations']]
            kinds=[]
            if fact and not fact['requested_vertical_motion_permitted'] and s['horizontal_distance']>1.:kinds.append('vertical_blocked_far')
            if any('must_brake_before_reversing_up'==r for r in reasons):kinds.append('reverse_velocity_brake')
            if fact and len(fact['permitted_vertical_controls'])==1:kinds.append('creep_only_vertical')
            if s['at_position']:kinds.append('position_reached')
            if s.get('doorway_phase',{}).get('phase')=='CROSS_OPENING':kinds.append('door_crossing')
            if s.get('active_detour'):kinds.append('active_bypass')
            kind=next((k for k in kinds if counts.get(k,0)<2),None)
            if kind:
                cases.append(dict(label=f'{kind}-{counts.get(kind,0)}',kind=kind,source=str(path),index=index,call=c));counts[kind]=counts.get(kind,0)+1
    if args.uniform:
        excluded=set()
        for prior in ('altitude','boundaries'):
            for c in json.loads(Path(f'results/px4-house-v39-{prior}-probe/cases.json').read_text()):
                excluded.add((c['source'],c['index']))
        cases=[]
        for trial in ('house_hidden-5311','house_reverse-5312'):
            path=Path('results/px4-house-v36-validation')/trial/'calls.jsonl'
            available=[]
            for index,line in enumerate(path.read_text().splitlines()):
                c=json.loads(line)
                if 'control_effects' in c['state'] and (str(path),index) not in excluded:
                    available.append((index,c))
            for quantile in range(20):
                index,c=available[round(quantile*(len(available)-1)/19)]
                cases.append(dict(label=f'{trial}-{quantile}',kind='uniform_unused_state',source=str(path),index=index,call=c))
    repeats=1 if args.uniform else 2
    (args.out/'cases.json').write_text(json.dumps(cases)+'\n')
    (args.out/'protocol.json').write_text(json.dumps({'cases':[{k:v for k,v in c.items() if k!='call'} for c in cases],
        'variants':['original','altitude'],'repeats':repeats,'scope':'Development replay audit of v39. No changes to running policy. Uniform mode samples 20 unused control states across each v36 flight after excluding earlier probe cases; boundary mode uses stratified early cases. Neither is an independent flight reliability estimate.'},indent=2)+'\n')
    gateway=JevGateway('openrouter',load_credential('openrouter',Path('.env')),journal=args.out/'calls.jsonl');rows=[]
    try:
        for case_index,case in enumerate(cases):
            for repeat in range(repeats):
                for variant in ('original','altitude') if (case_index+repeat)%2==0 else ('altitude','original'):
                    s=deepcopy(case['call']['state']);q=deepcopy(case['call']['questions']['selection'])
                    if variant=='altitude':s,q['criteria']=describe(s,q['criteria']);q['instructions']+=INSTRUCTIONS
                    r=gateway.choose(s,q['instructions'],q['criteria']);choice=r.get('choice');e=s['control_effects'].get(choice,{})
                    row=dict(case=case['label'],kind=case['kind'],variant=variant,repeat=repeat,choice=choice,violations=e.get('movement_violations'),travel=e.get('travel_effect'),turn=e.get('turn_effect'),error=r.get('error'))
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:
        gateway.close();(args.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
