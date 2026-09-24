"""Search attention from visible probe samples; unseen/occluded space stays unknown."""
from collections import defaultdict
import itertools

from experiments.legacy.jev_navigation import LOOK
from experiments.legacy.search_world import ROOMS
from experiments.legacy.sim import norm, ray_box, subtract, unit


def probes():
    result={}
    for room,bounds in ROOMS.items():
        x0=bounds["bounds_xyz"][0][0]
        for i,j,k in itertools.product(range(3),range(3),range(2)):
            region=f"{('west','middle','east')[i]}_{('south','middle','north')[j]}"
            for n,(dx,dy) in enumerate(itertools.product((-.35,.35),(-.55,.55))):
                name=f"{room}:{i}{j}{k}:{n}"
                result[name]={"room":room,"region":region,"height":"low" if k==0 else "high",
                              "xyz":(x0+1+i*2+dx,10/6+j*10/3+dy,.7 if k==0 else 2.5)}
    return result


PROBES=probes()


def in_view(point,position,look,range_m=3.5):
    delta=subtract(point,position)
    distance=norm(delta)
    return 1e-6<distance<=range_m and sum(a*b for a,b in zip(unit(delta),LOOK[look]))>=.5


def visible_probes(world,position,look):
    """Simulated camera observation, not a map passed to a planner."""
    observed=[]
    for name,probe in PROBES.items():
        point=probe["xyz"]
        if not in_view(point,position,look): continue
        delta=subtract(point,position)
        if any((hit:=ray_box(position,delta,box)) is not None and hit<1-1e-6 for box in world.obstacles): continue
        observed.append(name)
    return observed


class SearchAttention:
    def __init__(self):
        self.seen={}
        self.views=[]
        self.place_looks=defaultdict(set)

    def record(self,observed,position,look,now,room,objects=()):
        fresh=set(observed)-self.seen.keys()
        for name in observed: self.seen[name]=now
        cell=tuple(round(v) for v in position)
        repeated=look in self.place_looks[cell]
        self.place_looks[cell].add(look)
        event={"time":round(now,2),"room":room,"position":[round(v,2) for v in position],"look":look,
               "new_visible_samples":len(fresh),"visible_samples":len(observed),"repeat_view":repeated,"objects":list(objects)}
        self.views.append(event)
        return event

    def summary(self,room):
        regions=defaultdict(lambda:{"low":0,"high":0})
        total=seen=0
        for name,p in PROBES.items():
            if p["room"]!=room: continue
            total+=1
            regions[p["region"]][p["height"]]+=int(name in self.seen)
            seen+=int(name in self.seen)
        return {"seen_samples":seen,"total_samples":total,"fraction":round(seen/total,3),
                "regions":dict(regions),"meaning":"Each height has four representative samples per region. Seen means actually visible; 0 may be occluded or not looked at. This is not proof the region is empty."}

    def potential(self,position,look,room):
        # Geometry-free upper bound: no obstacle access, no predicted visibility oracle.
        return sum(name not in self.seen and p["room"]==room and in_view(p["xyz"],position,look)
                   for name,p in PROBES.items())

    def novelty(self,position,room):
        return {look:self.potential(position,look,room) for look in LOOK}
