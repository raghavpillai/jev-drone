"""Matched Jev search-memory comparisons on fixed, previously untested layouts."""
from jev_drone.evidence import snapshot_sources
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
from types import SimpleNamespace

from experiments.legacy.active_search import ActiveSearch
from jev_drone.gateway import JevGateway, load_credential

CASES = [
    ('occluded',502,'bookcase','bedroom'),
    ('near',503,'bed','bedroom'),
    ('absent',504,'bookcase','bedroom'),
    ('far',506,'bookcase','study'),
]


def run_one(key, out, case, seed, item, room, memory):
    args=SimpleNamespace(case=case,seed=seed,item=item,room=room,memory=memory,seconds=120.,budget=.15)
    gateway=JevGateway('openrouter',key)
    try:
        result=ActiveSearch(args,gateway).run()
    finally:
        gateway.close()
    name=f'{memory}-{case}-{item}-{seed}'
    (out/(name+'.json')).write_text(json.dumps(result,indent=2))
    return {k:result[k] for k in ('case','seed','item','room','memory_mode','status','simulation_seconds','role_counts','cost_usd','error')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    jobs=[(*case,memory) for case in CASES for memory in ('checklist','coverage')]
    random.Random(1901).shuffle(jobs)
    (args.out/'protocol.json').write_text(json.dumps({'jobs':jobs,'seconds':120,'workers':3,
        'depth':'ideal, 120 degree cone','semantics':'15% dropout, first two wooden sightings ambiguous',
        'model':'typesafe/jev-1.13','provider':'openrouter'},indent=2))
    key=load_credential('openrouter',args.key_file)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(run_one,key,args.out,*job):job for job in jobs}
        for future in as_completed(futures):
            print(json.dumps(future.result()),flush=True)


if __name__=='__main__':
    main()
