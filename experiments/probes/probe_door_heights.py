"""Offline planning experiment with alternate doorway heights from archived depth."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from jev_drone.gateway import JevGateway, load_credential
from experiments.probes.check_door_memory import recorded_cloud


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for cohort,trial in [('v36-validation','house_hidden-5311'),('v36-validation','house_reverse-5312'),('v37-validation','house_hidden-5411')]:
        directory=Path('results')/('px4-house-'+cohort)/trial
        result=json.loads((directory/'result.json').read_text());journal=[json.loads(x) for x in (directory/'calls.jsonl').read_text().splitlines()]
        frames=sorted((directory/'result-frames').glob('*.npz'));seen=set()
        for index,(logged,raw) in enumerate(zip(result['calls'],journal)):
            state=raw['state']
            if state.get('selected_task_kind')!='change_room':continue
            bad=[n for n,o in state['options'].items() if n.startswith('cross_') and o.get('observed_endpoint_unsafe')]
            if not bad or state['sensors']['room'] in seen:continue
            frame=max((f for f in frames if float(f.stem)<=logged['time']),key=lambda f:float(f.stem))
            cloud=recorded_cloud(frame,result['world']['hfov']);call=deepcopy(raw);alternatives={}
            def refresh(option):
                point=np.array(option['position']);gap=round(float(np.linalg.norm(cloud-point,axis=1).min())-.4,3)
                delta=point-state['sensors']['position_estimate']
                option.update(observed_goal_gap_m=gap,observed_endpoint_unsafe=gap<.08,
                    distance_m=round(float(np.linalg.norm(delta)),2),already_at_position=bool(np.linalg.norm(delta[:2])<=.35 and abs(delta[2])<=.15))
            for name,option in call['state']['options'].items():
                if 'position' in option:
                    refresh(option)
                    if name in call['questions']['selection']['criteria']:
                        call['questions']['selection']['criteria'][name]=('Observed endpoint unsafe. ' if option['observed_endpoint_unsafe'] else '')+json.dumps(option)
                if name.startswith(('approach_','cross_')) and 'position' in option:
                    higher=deepcopy(option);higher['position'][2]=1.9;higher['door']=name.split('_',1)[1]
                    higher['purpose']='Alternative doorway height, 1.9 m within the known 2.4 m opening. Requires Jev-controlled alignment and fresh depth; no route is certified.'
                    higher['observed_path_clearance_m']=None;refresh(higher);alternatives[name+'_high']=higher
            improved=[name+'_high' for name in bad if call['state']['options'][name]['observed_endpoint_unsafe'] and not alternatives[name+'_high']['observed_endpoint_unsafe']]
            call['state']['offline_endpoint_depth_frame']={'path':str(frame),'time':float(frame.stem),'lag_seconds':round(logged['time']-float(frame.stem),3),'scope':'Same archived depth for both arms, endpoint evidence only; not flight validation.'}
            cases.append(dict(label=f'{cohort}-{trial}-{index}',source=str(directory),index=index,improved_endpoints=improved,
                status='lower_endpoint_unsafe_high_clear' if improved else 'boundary_control',alternatives=alternatives,call=call))
            seen.add(state['sensors']['room'])
            if len(seen)>=2:break
    (a.out/'cases.json').write_text(json.dumps(cases)+'\n')
    (a.out/'protocol.json').write_text(json.dumps({'variants':['fixed_height','alternate_height'],'repeats':2,
        'cases':[{k:v for k,v in c.items() if k not in ('call','alternatives')} for c in cases],
        'scope':'Offline development only, no native policy changed. Every available adjacent-door approach/crossing receives a 1.9 m alternative. Both arms use identical archived depth endpoints. Opening dimensions are architectural knowledge, never hidden furniture. Positive endpoint gap does not prove a traversable path or achievable altitude.'},indent=2)+'\n')
    gateway=JevGateway('openrouter',load_credential('openrouter',Path('.env')),journal=a.out/'calls.jsonl');rows=[]
    try:
        for case in cases:
            for repeat in range(2):
                for variant in ('fixed_height','alternate_height') if repeat==0 else ('alternate_height','fixed_height'):
                    state=deepcopy(case['call']['state']);q=deepcopy(case['call']['questions']['selection'])
                    if variant=='alternate_height':
                        state['options'].update(case['alternatives'])
                        for name,o in case['alternatives'].items():q['criteria'][name]=('Observed endpoint unsafe. ' if o['observed_endpoint_unsafe'] else '')+json.dumps(o)
                        q['instructions']+='\nDoorway options may use different heights. A blocked endpoint at 1.3 m does not establish that every height is blocked. Assess each option using its depth evidence. A clear endpoint is not a clear route: the local Jev controller must align at that height and check clearance throughout the crossing. No crossing is automatic.'
                    r=gateway.choose(state,q['instructions'],q['criteria']);choice=r.get('choice')
                    row=dict(case=case['label'],status=case['status'],variant=variant,repeat=repeat,choice=choice,
                        improved_endpoint_selected=choice in case['improved_endpoints'],selected_observed_unsafe_endpoint=state['options'].get(choice,{}).get('observed_endpoint_unsafe',False),error=r.get('error'))
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:
        gateway.close();(a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
