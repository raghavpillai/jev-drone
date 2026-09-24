"""Compare near-goal axis facts against premature diagonal-path detours."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.control.axis_progress import describe, INSTRUCTIONS


OLD_RULE='When goal_path_status is BLOCKED, not at_position, and you are stopped, do NOT keep_controls forever:\nthat only keeps hovering. Select a detour with bypass_viable=true.'
NEW_RULE='When goal_path_status is BLOCKED and not at_position, first check near_goal_axis_progress. A permitted short body-axis correction can still approach the goal even when the exact diagonal is blocked. Consider that correction before a large bypass. If no such correction is available, do NOT keep hovering with zero controls: select a detour with bypass_viable=true.'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);cases=[];negatives={}
    sources=[('px4-house-v35-return','house_hidden-5111'),('px4-house-v30-survey','house_hidden-5111'),
             ('px4-house-v31-progress','house_search-5111')]
    for cohort,trial in sources:
        path=Path('results')/cohort/trial/'calls.jsonl';selected=set()
        for index,line in enumerate(path.read_text().splitlines()):
            c=json.loads(line);s=c['state']
            if 'control_effects' not in s:continue
            choice=c.get('answers',{}).get('selection',{}).get('choice');task=s['task']['name']
            updated,_=describe(s,c['questions']['selection']['criteria']);fact=updated.get('near_goal_axis_progress')
            if (len(selected)<2 and task not in selected and fact and s['goal_path_status'] in ('BLOCKED','UNKNOWN')
                    and choice not in fact['permitted_controls'] and choice and choice.startswith(('detour_','brake','keep_'))):
                cases.append({'label':cohort+'-'+task,'source':str(path),'index':index,'positive':True,'call':c});selected.add(task)
            negative=('at_position' if s['at_position'] else 'occupied_endpoint' if s.get('goal_endpoint_observed_occupied') else None)
            if negative and negative not in negatives:
                negatives[negative]={'label':negative,'source':str(path),'index':index,'positive':False,'call':c}
    cases.extend(negatives.values())
    (a.out/'cases.json').write_text(json.dumps(cases)+'\n')
    protocol={'cases':[{k:v for k,v in c.items() if k!='call'} for c in cases],
        'variants':['original','facts','facts_and_rule'],'repeats':2,
        'scope':'Issue-focused recorded states with a blocked/unknown diagonal and a permitted single-axis correction within 1 m. Identical actions, movement constraints and observations in all arms. Negative controls are already-arrived and occupied-goal states. Live-source requests are copied into cases.json. Not a flight/reliability result.'}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    gateway=JevGateway('openrouter',load_credential('openrouter',Path('.env')),journal=a.out/'calls.jsonl');rows=[]
    try:
        for case in cases:
            for repeat in range(2):
                for variant in (protocol['variants'] if repeat==0 else protocol['variants'][::-1]):
                    s=deepcopy(case['call']['state']);q=deepcopy(case['call']['questions']['selection'])
                    candidate,_=describe(s,q['criteria']);allowed=candidate.get('near_goal_axis_progress',{}).get('permitted_controls',[])
                    if variant!='original':
                        s,q['criteria']=describe(s,q['criteria']);q['instructions']+=INSTRUCTIONS
                    if variant=='facts_and_rule':
                        assert OLD_RULE in q['instructions'];q['instructions']=q['instructions'].replace(OLD_RULE,NEW_RULE)
                    result=gateway.choose(s,q['instructions'],q['criteria']);choice=result.get('choice')
                    row={'case':case['label'],'positive':case['positive'],'variant':variant,'repeat':repeat,'choice':choice,
                        'permitted_axis_progress':choice in allowed,'violations':s['control_effects'].get(choice,{}).get('movement_violations'),
                        'error':result.get('error')}
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:
        gateway.close();(a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
