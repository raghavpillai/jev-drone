"""Live Jev regression from a saved final-review boundary; not a new full episode."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.legacy.focused_search import FocusedSearch
from jev_drone.gateway import JevGateway, load_credential
from experiments.legacy.search_trials import prepare_output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    data=json.loads(args.episode.read_text())
    review=data['calls'][-1]
    assert review['role']=='mission' and review['state']['final_budget_review']
    protocol=json.loads((args.episode.parent/'protocol.json').read_text())
    job=next(j for j in protocol['jobs'] if j['id']==data['trial_id'])
    prepare_output(args.out,{'source':str(args.episode),'kind':'checkpoint regression','repeats':args.repeats})
    key=load_credential('openrouter',args.key_file)
    for repeat in range(args.repeats):
        gateway=JevGateway('openrouter',key,journal=args.out/f'replay-{repeat}.calls.jsonl')
        try:
            config=SimpleNamespace(**({'seconds':150.,'budget':.2}|job))
            experiment=FocusedSearch(config,gateway)
            sim=experiment.sim;t=review['sent_at']
            row=max((r for r in data['trace'] if r[0]<=t),key=lambda r:r[0])
            sim.time=row[0];sim.position=tuple(row[1:4]);sim.velocity=tuple(row[4:7])
            previous=next(c for c in reversed(data['calls'][:-1]) if c['role']=='control' and c.get('accepted'))
            sim.command(previous['action'],speed=previous.get('speed'))
            sim.command_expires=previous['sent_at']+previous['latency_seconds']+sim.command_lease
            sim.advance(t-sim.time)
            sim.trace=[];sim.record()
            experiment.memory=data['objects_observed'].copy()
            experiment.sightings=Counter({oid:2 if 'unclear from this view' in o['appearance'] else 3 for oid,o in experiment.memory.items()})
            experiment.entries=data['room_entries'].copy()
            experiment.last_room=review['state']['current_room']
            experiment.tasks=data['tasks'][:-1].copy()
            experiment.view_completions=Counter(t['choice'] for t in experiment.tasks if t['outcome']=='viewpoint_inspected')
            for view in data['views']:
                if view['time']<=t:
                    experiment.attention.place_looks[tuple(round(v) for v in view['position'])].add(view['look'])
            look=data['views'][-1]['look']
            experiment.pilot.look=look;experiment.pilot.look_history=[(0.,look)]
            experiment.deadline_reviewed=True
            result=experiment.run()
        finally:
            gateway.close()
        result.update(trial_id=f'final-review-checkpoint-{repeat}',checkpoint_time=t,
                      checkpoint_note='Restored from timestamp-rounded trace and saved observation memory. Replays only the last review, not the full mission.',
                      continuation_seconds=result['simulation_seconds']-t)
        (args.out/f'replay-{repeat}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result[k] for k in ('trial_id','status','continuation_seconds','cost_usd')}),flush=True)


if __name__=='__main__':main()
