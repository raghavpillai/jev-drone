"""Run a frozen list of live Jev experiments, saving every episode independently."""
from jev_drone.evidence import snapshot_sources
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
from types import SimpleNamespace

from experiments.legacy.active_search import ActiveSearch
from jev_drone.gateway import JevGateway, load_credential
from experiments.legacy.purposeful_search import PurposefulSearch
from experiments.legacy.focused_search import FocusedSearch


def prepare_output(out,protocol):
    out.mkdir(parents=True,exist_ok=False)
    snapshot_sources(out / "source", experiments=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))


def run_one(key,out,job):
    args=SimpleNamespace(**({'seconds':150.,'budget':.2}|job))
    gateway=JevGateway('openrouter',key,journal=out/(job['id']+'.calls.jsonl'))
    try:
        experiment_class={'purposeful':PurposefulSearch,'focused':FocusedSearch,'baseline':ActiveSearch}[job.get('engine','purposeful')]
        experiment=experiment_class(args,gateway)
        result=experiment.run()
    finally:
        gateway.close()
    result['trial_id']=job['id']
    pending=out/(job['id']+'.json.tmp')
    pending.write_text(json.dumps(result,indent=2))
    pending.replace(out/(job['id']+'.json'))
    return {'id':job['id'],**{k:result[k] for k in ('status','simulation_seconds','role_counts','cost_usd','error')},
            'coverage':{r:v['fraction'] for r,v in result['coverage'].items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    jobs=json.loads(args.protocol.read_text())
    assert len({j['id'] for j in jobs})==len(jobs)
    prepare_output(args.out,{'jobs':jobs,'seconds':150.,'budget':.2,'workers':args.workers})
    random.Random(2219).shuffle(jobs)
    key=load_credential('openrouter',args.key_file)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_one,key,args.out,job):job for job in jobs}
        for future in as_completed(futures):
            print(json.dumps(future.result()),flush=True)


if __name__=='__main__':main()
