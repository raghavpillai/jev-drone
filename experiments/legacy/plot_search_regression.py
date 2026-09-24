"""Visualize the actual trajectory and altitude for a diagnosed search regression."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np

PAIRS=[('Before: stale approach and conflicting completion rules',Path('results/search-v6-heldout/heldout-occluded-bookcase-712.json')),
       ('After: current approach and shared completion facts',Path('results/search-v8-screen/v8-bookcase-712.json'))]


def main():
    fig,axes=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'height_ratios':[3,1.2]})
    fig.patch.set_facecolor('#f5f2ea')
    for col,(label,path) in enumerate(PAIRS):
        d=json.loads(path.read_text());ax=axes[0,col];low=axes[1,col]
        ax.set_facecolor('#fcfbf6');low.set_facecolor('#fcfbf6')
        for obj in d['ground_truth']['objects']:
            x,y,z=obj['box']['lo'];xx,yy,zz=obj['box']['hi']
            if x<6 or x>=12:continue
            target=obj['id'] in d['eligible_target_ids']
            ax.add_patch(Rectangle((x,y),xx-x,yy-y,facecolor='#df9354' if target else '#d7d5ca',edgecolor='#a09f94'))
            if obj['kind']=='screen':ax.text((x+xx)/2,(y+yy)/2,'partition',ha='center',va='center',fontsize=8)
            if target:ax.text((x+xx)/2,yy+.18,'bookcase',ha='center',fontsize=8,color='#8b4d22')
        trace=np.asarray(d['trace'])
        in_room=trace[:,1]>=6
        room_trace=trace[in_room]
        points=room_trace[:,1:3]
        segments=np.stack([points[:-1],points[1:]],axis=1)
        lines=LineCollection(segments,cmap='viridis',norm=plt.Normalize(0,4),linewidth=2.3)
        lines.set_array(room_trace[:-1,3]);ax.add_collection(lines)
        ax.scatter(*points[0],c='#286e63',s=35,zorder=4,label='entry')
        ax.scatter(*points[-1],c='#a33c39',s=55,marker='x',zorder=4,label='finish')
        ax.set(xlim=(6,12),ylim=(0,10),aspect='equal',xlabel='x / metres',ylabel='y / metres')
        ax.set_title(f"{label}\n{d['status'].replace('_',' ')} · {d['simulation_seconds']:.1f}s",fontsize=11,pad=15)
        low.plot(trace[:,0],trace[:,3],color='#276c65',linewidth=1.8)
        low.axhline(1.8,color='#95958c',linestyle='--',linewidth=1,label='partition height')
        low.set(xlabel='Simulation time / seconds',ylabel='Height / metres',ylim=(0,4),xlim=(0,150))
        low.legend(frameon=False,fontsize=8,loc='upper right')
        for a in (ax,low):
            for spine in a.spines.values():spine.set_color('#b4b1a7')
            a.tick_params(labelsize=8)
    fig.suptitle('The drone saw the bookcase, but could not finish the approach',x=.06,ha='left',fontsize=16,y=.99)
    fig.text(.06,.025,'Same room layout, seed 712. Path color indicates flight height (dark: low, yellow: high). Geometry shown here is evaluator-only.',fontsize=9,color='#55584f')
    fig.tight_layout(rect=[0,.055,1,.95],h_pad=2)
    fig.savefig('report/search-regression.png',dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__=='__main__':main()
