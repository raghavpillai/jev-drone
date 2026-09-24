"""Ablate region visibility facts on recorded native planner requests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.visibility_memory import VisibilityMemory, INSTRUCTIONS
from jev_drone.world.layout import HOUSE


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    path=Path('results/px4-house-v30-diagnostics/house_search-5111')
    calls=[json.loads(x) for x in (path/'calls.jsonl').read_text().splitlines()]
    candidates=[c for c in calls if c['state'].get('selected_task_kind')=='search'][:5]
    cases=[];memory=VisibilityMemory(HOUSE);files=iter(sorted((path/'result-frames').glob('*.npz')));pending=next(files,None)
    for index,call in enumerate(candidates):
        stamp=call['state']['sensors'].get('front_frame_time',0)
        while pending and float(pending.stem)<=stamp:
            with np.load(pending) as data:
                for i,name in enumerate(data['camera_names']):
                    if name!='east':continue
                    frame=SimpleNamespace(camera_name=name,time=data['times'][i],depth=data['depth'][i],
                        origin=data['origins'][i],forward=data['forward'][i],right=data['right'][i],up=data['up'][i],focal=data['focal'][i])
                    memory.observe([frame])
            pending=next(files,None)
        cases.append({'index':index,'time':stamp,'call':call,'room_visibility':memory.summary(call['state']['sensors']['room'])})
    (a.out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (a.out/'protocol.json').write_text(json.dumps({'source':str(path),'repeats':2,'variants':['baseline','instructions','visibility'],
        'scope':'First five recorded search decisions, with sparse saved front-depth memory at request time. Choice differences only, no claimed flight improvement. No geometry or target truth enters memory; identical option universe in all arms.'},indent=2)+'\n')
    gateway=JevGateway('openrouter',load_credential('openrouter',Path('.env')),journal=a.out/'calls.jsonl');rows=[]
    try:
        for case in cases:
            for repeat in range(2):
                variants=['baseline','instructions','visibility']
                if repeat:variants.reverse()
                for variant in variants:
                    state=deepcopy(case['call']['state']);q=deepcopy(case['call']['questions']['selection'])
                    if variant!='baseline':q['instructions']+=INSTRUCTIONS
                    if variant=='visibility':state['room_visibility']=case['room_visibility']
                    response=gateway.choose(state,q['instructions'],q['criteria'])
                    row=dict(index=case['index'],time=case['time'],repeat=repeat,variant=variant,choice=response.get('choice'),error=response.get('error'))
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:
        gateway.close();(a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
