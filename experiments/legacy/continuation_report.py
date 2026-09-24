"""Summarize continued search trials and independently verify reported successes."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import re

from experiments.legacy.jev_navigation import LOOK
from experiments.legacy.search_world import Home, Object, evaluate_report, visible_objects
from experiments.legacy.sim import Box, Scenario, Simulation, norm, subtract


def paths():
    return sorted(p for folder in Path('results').glob('search-v*') for p in folder.glob('*.json') if p.name!='protocol.json')


def verify(data):
    if data['status']!='success':return None
    raw=data['world']
    box=lambda b:Box(tuple(b['lo']),tuple(b['hi']))
    world=Scenario(raw['name'],raw['seed'],tuple(raw['start']),tuple(raw['goal']),tuple(box(b) for b in raw['obstacles']),tuple(raw['size']))
    objects=tuple(Object(o['id'],o['kind'],box(o['box']),o['appearance']) for o in data['ground_truth']['objects'])
    report=data['calls'][-1]
    choice=report['choice'];oid=choice.removeprefix('report_found_')
    assert oid in data['eligible_target_ids']
    state=report['state'];evidence=next(o for o in state.get('objects',state.get('observed_objects',[])) if o['id']==oid)
    assert evidence['visible_now'] and 'unclear from this view' not in evidence['appearance']
    home=Home(world,objects,oid)
    sim=Simulation(world)
    sim.position=tuple(data['trace'][-1][1:4]);sim.velocity=tuple(data['trace'][-1][4:7])
    look=data['views'][-1]['look']
    sensed=visible_objects(home,sim.position,view_direction=LOOK[look])
    result=evaluate_report(choice,home,sim,sensed)
    assert result=='success',(data['trial_id'],result)
    return 'verified'


def stationary_task_streak(tasks):
    anchor=None
    count=maximum=0
    for task in tasks:
        position=task.get('position')
        if position is None:continue
        if anchor is None or norm(subtract(position,anchor))>=.15:
            anchor=position
            count=1
        else:
            count+=1
        maximum=max(maximum,count)
    return maximum


def control_violations(calls):
    counts=Counter()
    for call in calls:
        if call['role']!='control' or not call.get('accepted') or call.get('action') not in LOOK:
            continue
        action=call['action']
        state=call['state']
        common=json.loads(state.splitlines()[1])
        reading=re.search(rf'(?m)^{action}: (OPEN|BLOCKED|UNKNOWN)(?:, clearance ([\d.]+) m)?',state)
        slow=call.get('speed',1.)<=.3
        enough_depth=(reading and reading[2] is not None and float(reading[2])>=.45) if slow else (reading and reading[1]=='OPEN')
        if not enough_depth:counts['translation_without_sufficient_depth']+=1
        if common['goal']['inside_arrival_radius']:counts['translation_after_arrival']+=1
        motion=common.get('flight_status')
        if motion and not motion['stopped'] and motion['current_motion_direction']!=action:
            counts['direction_change_before_stopping']+=1
        if slow and motion and not motion['slow_speed_ready']:
            counts['slow_command_while_moving_fast']+=1
    return dict(counts)


def summarize(path):
    data=json.loads(path.read_text())
    calls=data['calls'];control=[c for c in calls if c['role']=='control']
    latency=sorted(c['latency_seconds'] for c in control)
    coverage={r:v['fraction'] for r,v in data['coverage'].items()}
    target_present=bool(data['eligible_target_ids'])
    accepted=(data['status']=='success') if target_present else (
        data['status']=='search_incomplete' and data['room'] in coverage and coverage[data['room']]>=.85)
    counts=Counter(t['choice'] for t in data['tasks'] if not t['choice'].startswith('report_'))
    stationary=stationary_task_streak(data['tasks'])
    violations=control_violations(calls)
    target_sightings=[v['time'] for v in data['views'] if set(v['objects']) & set(data['eligible_target_ids'])]
    entry_looks=[{'room':entry['room'],'time':entry['time'],
                  'first_six_seconds_looks':sorted({v['look'] for v in data['views'] if v['room']==entry['room'] and entry['time']<=v['time']<=entry['time']+6})}
                 for entry in data['room_entries']]
    return {'source':str(path),'phase':path.parent.name,'id':data['trial_id'],
        **{k:data[k] for k in ('case','seed','item','room','variation','experiment','status','simulation_seconds','path_metres','cost_usd')},
        'target_present':target_present,'acceptance_pass':accepted,'report_verification':verify(data),
        'stationary_task_streak':stationary,'completion_and_loop_check_pass':accepted and stationary<=10,
        'control_rule_violations':violations,
        'strict_pass':accepted and stationary<=10 and not violations,
        'completed_viewpoints':sum(t['outcome']=='viewpoint_inspected' for t in data['tasks']),
        'room_entry_looks':entry_looks,'first_target_instance_sighting_s':min(target_sightings) if target_sightings else None,
        'coverage':coverage,'maximum_altitude_m':round(max(row[3] for row in data['trace']),2),
        'repeated_tasks':{k:v for k,v in counts.items() if v>1},'stalled_tasks':sum(t['outcome'].startswith('stalled') for t in data['tasks']),
        'calls':len(calls),'roles':dict(Counter(c['role'] for c in calls)),
        'model_ids':sorted({c.get('model','') for c in calls}),
        'api_errors':sum(bool(c.get('error')) for c in calls),
        'unpriced_api_errors':sum(bool(c.get('error')) and 'cost' not in c.get('usage',{}) for c in calls),
        'stale_controls':sum(c['latency_seconds']>data['physics']['command_lease_s'] for c in control),
        'median_control_latency_ms':round(statistics.median(latency)*1000,1) if latency else None,
        'p95_control_latency_ms':round(latency[min(len(latency)-1,int(len(latency)*.95))]*1000,1) if latency else None,
        'inference_capacity_hz':round(len(control)/sum(latency),2) if latency else None,
        'control_hz':round(len(control)/sum(max(.2,t) for t in latency),2) if latency else None,
        'mission_control_hz':round(len(control)/data['simulation_seconds'],2) if latency else None,
        'accepted_mission_control_hz':round(sum(bool(c.get('accepted')) for c in control)/data['simulation_seconds'],2) if latency else None}


def main():
    results=[summarize(p) for p in paths()]
    phases={}
    for phase in sorted({r['phase'] for r in results}):
        rows=[r for r in results if r['phase']==phase]
        present=[r for r in rows if r['target_present']]
        absent=[r for r in rows if not r['target_present']]
        phases[phase]={'episodes':len(rows),'target_successes':sum(r['status']=='success' for r in present),'target_trials':len(present),
            'absence_search_passes':sum(r['acceptance_pass'] for r in absent),'absence_trials':len(absent),
            'completion_and_loop_passes':sum(r['completion_and_loop_check_pass'] for r in rows),
            'strict_passes':sum(r['strict_pass'] for r in rows),
            'episodes_with_control_violations':sum(bool(r['control_rule_violations']) for r in rows),
            'api_errors':sum(r['api_errors'] for r in rows),
            'statuses':dict(Counter(r['status'] for r in rows)),'calls':sum(r['calls'] for r in rows),'cost_usd':sum(r['cost_usd'] for r in rows)}
    report={'phases':phases,'episodes':results}
    Path('report/search-continuation-results.json').write_text(json.dumps(report,indent=2))
    columns=['phase','id','item','room','variation','status','simulation_seconds','path_metres',
             'acceptance_pass','stationary_task_streak','completion_and_loop_check_pass','strict_pass','calls','cost_usd','control_hz']
    with Path('report/search-continuation-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(results)
    print(json.dumps(phases,indent=2))


if __name__=='__main__':main()
