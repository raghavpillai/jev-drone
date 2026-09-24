"""Offline coverage audit using saved front depth frames, never a controller input.

Measures sampled free-space visibility at three heights, not target recognition
or proof of absence. Scene geometry is used only to exclude solid grid samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from jev_drone.planning.visibility_memory import sample_depth


def visible_samples(samples, depth, origin, forward, right, up, focal):
    return sample_depth(samples, depth, origin, forward, right, up, focal)[0]


def analyze(path,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    result=json.loads(path.read_text());objective=result['world']['mission'][0]
    room=objective['room'];lo,hi=map(np.array,result['world']['rooms'][room])
    raw=[json.loads(x) for x in path.with_name('calls.jsonl').read_text().splitlines()]
    calls=[{**log,'state':source['state']} for log,source in zip(result['calls'],raw)]
    entered=next((c['time'] for c in calls if c['state']['sensors']['room']==room),None)
    seen=next((c['time'] for c in calls if c['state']['sensors']['room']==room
               and any(d['marker']==objective['marker'] for d in c['state']['sensors']['detections'])),None)
    start=entered if entered is not None else result['sim_seconds']
    end=seen if seen is not None else result['sim_seconds']
    grid=np.array([[x,y,z] for z in (.6,1.6,2.4)
        for y in np.arange(lo[1]+.25,hi[1],.25) for x in np.arange(lo[0]+.25,hi[0],.25)])
    solid=np.zeros(len(grid),dtype=bool)
    for box in result['world']['geometry']:
        solid|=(np.abs(grid-box['center'])<=np.array(box['half'])+.04).all(axis=1)
    samples=grid[~solid];first=np.full(len(samples),np.nan);frames=0;headings=[];timeline=[]
    for file in sorted(path.with_name('result-frames').glob('*.npz')):
        nominal=float(file.stem)
        if not start-.65<=nominal<=end:continue
        with np.load(file) as data:
            matches=np.flatnonzero(data['camera_names']=='east')
            if not len(matches):continue
            i=matches[0];origin=data['origins'][i];stamp=float(data['times'][i])
            if not start-.65<=stamp<=end or not ((origin[:2]>=lo[:2])&(origin[:2]<=hi[:2])).all():continue
            visible=visible_samples(samples,data['depth'][i],origin,data['forward'][i],data['right'][i],data['up'][i],data['focal'][i])
            fresh=visible&~np.isfinite(first);first[fresh]=stamp
            frames+=1;timeline.append({'time':stamp,'new_samples':int(fresh.sum()),'seen_samples':int(np.isfinite(first).sum())})
            headings.append((stamp,origin.copy(),data['forward'][i].copy()))
    metrics=[]
    fig,axes=plt.subplots(1,3,figsize=(15,5.6),sharex=True,sharey=True)
    for ax,z in zip(axes,(.6,1.6,2.4)):
        mask=samples[:,2]==z;known=mask&np.isfinite(first);unknown=mask&~np.isfinite(first)
        for box in result['world']['geometry']:
            center,half=np.array(box['center']),np.array(box['half'])
            if abs(center[2]-z)>half[2] or box['name'] in ('floor','ceiling'):continue
            ax.add_patch(Rectangle(center[:2]-half[:2],2*half[0],2*half[1],facecolor='#bfc2c5',alpha=.6))
        ax.scatter(*samples[unknown,:2].T,c='#dadfe5',s=8,marker='s')
        points=ax.scatter(*samples[known,:2].T,c=first[known]-start,s=8,marker='s',cmap='viridis',vmin=0,vmax=max(end-start,1))
        nearby=[h for h in headings if abs(h[1][2]-z)<.6]
        if nearby:
            stride=max(1,len(nearby)//18)
            for _,pose,forward in nearby[::stride]:
                ax.arrow(pose[0],pose[1],forward[0]*.3,forward[1]*.3,width=.012,head_width=.1,color='#c76533',alpha=.75)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='x (m)')
        count=int(known.sum());total=int(mask.sum());fraction=count/total if total else 0
        milestones = {}
        observed_times = np.sort(first[known])
        for threshold in (80,90,95,99):
            required = int(np.ceil(total*threshold/100))
            milestones[f'seconds_to_{threshold}pct'] = max(0.,float(observed_times[required-1]-start)) if required and count>=required else None
        metrics.append({'height_m':z,'visible_samples':count,'free_samples':total,'fraction':fraction,**milestones})
        ax.set_title(f'{z:.1f} m plane\n{fraction:.1%} sampled free space seen',fontsize=11)
    axes[0].set_ylabel('y (m)')
    fig.colorbar(points,ax=axes.ravel().tolist(),shrink=.6,label='Seconds after room entry when first seen',pad=.02)
    fig.suptitle(f"{path.parent.parent.name} / {result['id']}\n{room} search phase · {end-start:.1f} s · {frames} saved front frames")
    fig.text(.03,.04,'Gray squares remain unobserved in saved front depth; orange arrows show camera poses. This is not object-recognition coverage or proof of absence.',fontsize=9)
    out.mkdir(parents=True,exist_ok=True);fig.savefig(out/'coverage.png',dpi=180);plt.close(fig)
    summary={'source':str(path),'room':room,'start':start,'end':end,'saved_front_frames':frames,'planes':metrics,
        'timeline':timeline,'method':'0.25 m grid; solid cells excluded using evaluator geometry. A point is visible only inside a saved FRONT frame with >=3 finite pixels in its 3x3 patch and 20th percentile depth >= axial range + .035 m. Sparse saved frames are a visibility sample, not recognition evidence or proof of absence.'}
    (out/'coverage.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='timeline'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();analyze(args.episode,args.out)
