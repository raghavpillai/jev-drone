"""Paired archived-state Jev probes; no action masking or automatic correction."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from jev_drone.gateway import JevGateway, load_credential
from experiments.representations.control_presentation import describe, INSTRUCTIONS
from experiments.representations.altitude_access import describe as altitude_access, INSTRUCTIONS as ALTITUDE_INSTRUCTIONS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--violations-only',action='store_true');p.add_argument('--altitude',action='store_true');p.add_argument('--native-order',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    cases=[]
    sources=[('v34-full','house_hidden-5111'),('v36-validation','house_reverse-5312'),
             ('v37-validation','house_hidden-5411'),('v38-validation','house_reverse-5512')]
    for cohort,trial in sources:
        path=Path('results')/('px4-house-'+cohort)/trial
        result=json.loads((path/'result.json').read_text())
        violations={v['time'] for v in result['violations']}
        calls=[json.loads(line) for line in (path/'calls.jsonl').read_text().splitlines()]
        selected=0
        assert len(calls)==len(result['calls']), 'Request/result journal mismatch'
        for i,(c,logged) in enumerate(zip(calls,result['calls'])):
            if logged.get('time') not in violations or 'control_effects' not in c['state']:continue
            if selected>=2:break
            cases.append(dict(label=f'{cohort}-{selected}',kind='historical_violation',source=str(path/'calls.jsonl'),index=i,call=c));selected+=1
    path=Path('results/px4-house-v36-validation/house_hidden-5311/calls.jsonl')
    wanted={'forward_slow','strafe_left_slow','strafe_right_slow','up','down','pan_left','pan_right','brake'}
    for i,line in enumerate(path.read_text().splitlines()):
        c=json.loads(line);s=c['state'];choice=c.get('answers',{}).get('selection',{}).get('choice')
        if a.violations_only or 'control_effects' not in s or choice not in wanted:continue
        if s['control_effects'][choice]['movement_violations']:continue
        cases.append(dict(label='nominal-'+choice,kind='nominal',source=str(path),index=i,call=c));wanted.remove(choice)
        if not wanted:break
    variants=['original','altitude'] if a.native_order else ['original','altitude','altitude_summary'] if a.altitude else ['original','criteria','summary'];repeats=2
    (a.out/'cases.json').write_text(json.dumps(cases)+'\n')
    protocol={'cases':[{k:v for k,v in c.items() if k!='call'} for c in cases],'variants':variants,'repeats':repeats,'native_instruction_order':a.native_order,
        'scope':'Development replay probe. All controls and physical effects remain available/unchanged. Compare clearance violations and nominal progress without treating any particular control as the only correct answer. Historical v36-v38 trials are development sources, not holdout data for this change.'}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    gateway=JevGateway('openrouter',load_credential('openrouter',Path('.env')),journal=a.out/'calls.jsonl');rows=[]
    try:
        for case in cases:
            for repeat in range(repeats):
                for variant in variants if repeat==0 else variants[::-1]:
                    s=deepcopy(case['call']['state']);q=deepcopy(case['call']['questions']['selection'])
                    if variant in ('criteria','summary','altitude_summary'):
                        s,q['criteria']=describe(s,q['criteria'],summary=variant!='criteria')
                        if variant!='criteria':q['instructions']+=INSTRUCTIONS
                    if variant.startswith('altitude'):
                        s,q['criteria']=altitude_access(s,q['criteria'])
                        if a.native_order:
                            anchor='\nRead mission_progress before refining the waypoint.'
                            assert anchor in q['instructions']
                            q['instructions']=q['instructions'].replace(anchor,ALTITUDE_INSTRUCTIONS+anchor,1)
                        else:q['instructions']+=ALTITUDE_INSTRUCTIONS
                    result=gateway.choose(s,q['instructions'],q['criteria']);choice=result.get('choice');effect=s['control_effects'].get(choice,{})
                    row=dict(case=case['label'],kind=case['kind'],variant=variant,repeat=repeat,choice=choice,
                        violations=effect.get('movement_violations'),travel=effect.get('travel_effect'),turn=effect.get('turn_effect'),
                        controls=effect.get('resulting_controls'),latency_seconds=result['latency_seconds'],error=result.get('error'))
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:
        gateway.close();(a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
