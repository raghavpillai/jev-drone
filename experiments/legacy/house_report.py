"""Render semantic-room navigation runs without inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from experiments.legacy.house import WAYPOINTS


def main():
    names=["house-bedroom", "house-bookshelf-roomlabel"]
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    summary=[]
    for ax,name in zip(axes,names):
        e=json.loads((Path("results")/name/"episode.json").read_text())
        world=e["world"]
        for box in world["obstacles"]:
            lo,hi=box["lo"],box["hi"]
            ax.add_patch(Rectangle((lo[0],lo[1]),hi[0]-lo[0],hi[1]-lo[1],color="#cbd1d6"))
        ax.text(2.3,11.5,"LIVING ROOM",fontsize=10,color="#4b5663")
        ax.text(9,11.5,"BEDROOM",fontsize=10,color="#4b5663")
        ax.text(1.2,10.6,"bookshelf",fontsize=8)
        trace=e["trace"]
        ax.plot([p[1] for p in trace],[p[2] for p in trace],color="#187d67",linewidth=2,label="Executed local movements")
        ax.scatter(world["start"][0],world["start"][1],s=55,color="#263546",zorder=5)
        ax.scatter(world["goal"][0],world["goal"][1],s=160,marker="*",color="#1673b4",zorder=5)
        sequence=[]
        for p in e["plans"]:
            if p["kind"]=="waypoint" and (not sequence or sequence[-1]!=p["choice"]):
                sequence.append(p["choice"])
        for i,node in enumerate(sequence,1):
            xy=world["goal"] if node=="destination" else WAYPOINTS[node]["xyz"]
            ax.annotate(str(i),xy[:2],xytext=(6,8),textcoords="offset points",fontsize=12,fontweight="bold",color="#a35215")
        ax.set(xlim=(0,14),ylim=(0,12),xlabel="x / metres",ylabel="y / metres")
        ax.set_aspect("equal")
        ax.grid(alpha=.12)
        ax.set_title(e["mission"]+f"\n{e['status']} · {e['simulation_seconds']:.1f}s",loc="left",fontsize=12)
        summary.append({"run":name,"status":e["status"],"seconds":e["simulation_seconds"],"global_waypoints":sequence,
                        "global_calls":len(e["plans"]),"local_calls":len(e["decisions"]),
                        "accepted_local_hz":sum(d["accepted"] for d in e["decisions"])/e["simulation_seconds"],"cost_usd":e["cost_usd"]})
    fig.suptitle("Jev global + local navigation\nNumbers: selected global waypoints · Green: executed local trajectory · Gray: solid objects",fontsize=15)
    Path("report").mkdir(exist_ok=True)
    fig.savefig("report/house-navigation.png",dpi=140)
    plt.close(fig)
    Path("report/house-summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
