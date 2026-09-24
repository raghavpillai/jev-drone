"""Summarize recorded experiments; plots display evaluator geometry, never model input."""
from collections import Counter
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def files():
    return sorted(Path('results').glob('active-search-v2-*/episode.json')) + sorted(Path('results/active-search-v2-suite').glob('*-*-*.json'))


def summarize(path):
    data=json.loads(path.read_text())
    calls=data['calls']
    control=[c for c in calls if c['role']=='control']
    latencies=sorted(c['latency_seconds'] for c in control)
    tasks=Counter(t['choice'] for t in data['tasks'] if not t['choice'].startswith('report_'))
    entries=[]
    for entry in data['room_entries']:
        entry=dict(entry)
        views=[v for v in data['views'] if v['room']==entry['room'] and entry['time']<=v['time']<=entry['time']+6]
        entry['first_6s_look_directions']=sorted({v['look'] for v in views})
        entries.append(entry)
    return {'source':str(path), **{k:data[k] for k in ('case','seed','item','room','memory_mode','status','simulation_seconds','cost_usd','path_metres')},
        'calls':len(calls),'role_counts':data['role_counts'],'model_ids':sorted({c.get('model','') for c in calls}),
        'control_hz':round(len(control)/sum(latencies),2) if latencies else None,
        'control_latency_median_ms':round(statistics.median(latencies)*1000,1) if latencies else None,
        'control_latency_p95_ms':round(latencies[min(len(latencies)-1,int(len(latencies)*.95))]*1000,1) if latencies else None,
        'stale_control_responses':sum(c['latency_seconds']>.8 for c in control),
        'api_errors':sum(bool(c.get('error')) for c in calls),
        'coverage':{r:v['fraction'] for r,v in data['coverage'].items()},
        'repeated_tasks':{k:v for k,v in tasks.items() if v>1},
        'room_entries':entries,'tasks':data['tasks'],
        'zero_gain_repeat_views':sum(v['new_visible_samples']==0 and v['repeat_view'] for v in data['views']),
        'observations':len(data['views'])}


def plot(results):
    seeds=sorted({r['seed'] for r in results})
    fig,axes=plt.subplots(len(seeds),2,figsize=(14,3.5*len(seeds)),squeeze=False)
    fig.patch.set_facecolor('#f4f2ec')
    for row,seed in enumerate(seeds):
        for col,mode in enumerate(('checklist','coverage')):
            ax=axes[row,col]
            match=next((r for r in results if r['seed']==seed and r['memory_mode']==mode),None)
            if match is None:
                ax.set_visible(False); continue
            d=json.loads(Path(match['source']).read_text())
            ax.set_facecolor('#fcfbf7')
            targets={o['id']:o for o in d['ground_truth']['objects']}
            for box in d['world']['obstacles']:
                x,y,z=box['lo']; xx,yy,zz=box['hi']
                color='#d9d6cd' if zz<3 else '#777d7b'
                ax.add_patch(Rectangle((x,y),xx-x,yy-y,facecolor=color,edgecolor='#b5b7ae',alpha=.9))
            target=targets.get(d['ground_truth']['target_id'])
            if target:
                x,y,_=target['box']['lo']; xx,yy,_=target['box']['hi']
                ax.add_patch(Rectangle((x,y),xx-x,yy-y,facecolor='#de9357',edgecolor='#945623',linewidth=1.3))
            trace=d['trace']
            ax.plot([t[1] for t in trace],[t[2] for t in trace],color='#276a66',linewidth=1.6)
            ax.scatter(trace[0][1],trace[0][2],color='#276a66',marker='o',s=25,zorder=4)
            ax.scatter(trace[-1][1],trace[-1][2],color='#a53837',marker='x',s=40,zorder=4)
            for i,task in enumerate(d['tasks']):
                if task['choice'].startswith(('look_','inspect_')):
                    x,y,_=task['position']; ax.scatter(x,y,s=75,facecolors='none',edgecolors='#d3993c',zorder=5)
            for x,name in ((3,'LIVING'),(9,'BEDROOM'),(15,'STUDY')):
                ax.text(x,10.35,name,ha='center',fontsize=8,color='#696f69')
            label='Viewpoint memory' if mode=='checklist' else 'Visible-region memory'
            condition={'absent':' · target absent','occluded':' · partition'}.get(d['case'],'')
            ax.set_title(f"{label} · {d['item']} in {d['room'].replace('_',' ')}{condition}\n{d['status'].replace('_',' ')} · {d['simulation_seconds']:.1f}s · {d['path_metres']:.1f}m",loc='left',fontsize=10,pad=17)
            ax.set(xlim=(0,18),ylim=(0,10),aspect='equal')
            ax.set_xticks([0,6,12,18]); ax.set_yticks([0,5,10]); ax.tick_params(labelsize=8)
            for spine in ax.spines.values(): spine.set_color('#b5b7ae')
    fig.suptitle('Jev searches unfamiliar rooms: matched memory comparisons',fontsize=17,x=.07,ha='left',y=.995)
    fig.text(.07,.009,'Green: flight path. Gold rings: explicit look/inspection tasks. Orange: evaluator-only target. Red ×: final pose. Top view; vertical motion is simulated.',fontsize=9,color='#555c56')
    fig.tight_layout(rect=[0,.03,1,.975],h_pad=2)
    fig.savefig('report/active-search-paths.png',dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    results=[summarize(p) for p in files()]
    Path('report/active-search-results.json').write_text(json.dumps(results,indent=2))
    plot(results)
    for r in results:
        print(json.dumps({k:r[k] for k in ('seed','item','room','memory_mode','status','simulation_seconds','coverage','repeated_tasks','control_hz')}))


if __name__=='__main__': main()
