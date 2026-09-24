"""Check a doorway-only release annotation against braking and arrival controls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.control.doorway_phase import describe as phase, INSTRUCTIONS as PHASE
from experiments.representations.doorway_progress import describe, INSTRUCTIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    journal=Path('results/px4-house-v27-furnished/house_hidden-5111/calls.jsonl')
    calls=[json.loads(x) for x in journal.read_text().splitlines()]
    eligible=[]
    for c in calls:
        if 'control_effects' not in c['state']:continue
        s,_=phase(c['state'],c['questions']['selection']['criteria']);a,_=describe(s,c['questions']['selection']['criteria'])
        if a.get('aligned_crossing_waiting_for_movement') and c.get('answers',{}).get('selection',{}).get('choice')=='stop_pan':eligible.append(c)
    cases=[('stationary',c) for c in eligible[::max(1,len(eligible)//3)][:3]]
    course=[json.loads(x) for x in Path('results/px4-house-v28-control/controller-4910/calls.jsonl').read_text().splitlines()]
    for kind,predicate,count in [('braking',lambda s:s.get('neutral_controls_still_braking'),2),('arrival',lambda s:s.get('at_position'),1)]:
        candidates=[c for c in course if predicate(c['state'])]
        cases.extend((kind,c) for c in candidates[::max(1,len(candidates)//count)][:count])
    blocked=[c for c in calls if c['state'].get('doorway_alignment') and not describe(*phase(c['state'],c['questions']['selection']['criteria']))[0].get('aligned_crossing_waiting_for_movement')]
    if blocked:cases.append(('ineligible_crossing',blocked[-1]))
    gateway=JevGateway('openrouter',load_credential('openrouter',Path('.env')),journal=args.out/'calls.jsonl');rows=[]
    try:
        for index,(kind,call) in enumerate(cases):
            for repeat in range(2):
                for variant in (('phase_only','crossing_progress') if repeat==0 else ('crossing_progress','phase_only')):
                    state=deepcopy(call['state']);q=deepcopy(call['questions']['selection'])
                    state,q['criteria']=phase(state,q['criteria'])
                    if 'doorway_phase' in state:q['instructions']+=PHASE
                    if variant=='crossing_progress':
                        state,q['criteria']=describe(state,q['criteria'])
                        if state.get('aligned_crossing_waiting_for_movement'):q['instructions']+=INSTRUCTIONS
                    r=gateway.choose(state,q['instructions'],q['criteria']);choice=r.get('choice');e=state['control_effects'].get(choice,{})
                    row=dict(index=index,kind=kind,repeat=repeat,variant=variant,choice=choice,violations=e.get('movement_violations'),
                        forward_toward=e.get('travel_effect',{}).get('forward')=='toward_goal',error=r.get('error'))
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:
        gateway.close();(args.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
