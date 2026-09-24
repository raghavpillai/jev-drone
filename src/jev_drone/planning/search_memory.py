"""Bounded memory of camera poses near known-room search viewpoints."""
import math

import numpy as np

from jev_drone.world.layout import SMALL


INSTRUCTIONS = """
Read search_phase. First reach the requested room through connected doorways.
Outside that room, viewpoints can only help transit around obstacles; this is
not a search of the requested room. Inside it, search requires looking as well
as moving. Each view task means reach its position THEN face look_position into
the room; arriving while staring at the nearest wall is not that survey.
Start with useful nearby low viewpoints and inward views. Add higher viewpoints
when a different height is useful and depth supports it, not simply because a
higher coordinate is offered. After a useful inward view, use local_survey and
new_heading_sectors to look toward headings not yet observed from here. Avoid
repeating a satisfied look with zero new sectors while other useful views remain.
Then change position to see behind furniture/partitions, including the opposite
side of the room. Do not endlessly turn in one place or abandon a position before
looking into the room. You choose every viewpoint and look; no sweep is automatic.
observed_near_viewpoint only means the camera passed nearby. surveyed_heading_sectors
requires a fresh front frame while settled, and inward_view_observed records a
view toward the room interior. These are camera-pose coverage, NOT proof that
occluded space was searched or that an item is absent. Even all headings at one
position can miss a target behind furniture. Low/high viewpoints stay separate.
Once the requested target is observed, approach that lead or select an intermediate
viewpoint to get around its occluder; do not restart blind coverage. Report only
when inspection_ready. Do not give_up just because several views found nothing
while useful unobserved views and time remain. Never infer absence from this memory.
"""


def heading_sectors(yaw):
    # Central 90 degrees of the 137.5-degree front image, avoiding its edges.
    return {heading for heading in range(0,360,45) if abs((heading-yaw+180)%360-180)<=45}


class SearchMemory:
    def __init__(self, layout=SMALL):
        self.layout = layout
        self.viewpoints = {name:{**view,'headings':set(),'surveyed':set(),
                                'inward':False,'last_observed':None}
                           for name,view in layout.search_viewpoints().items()}
        self.stations = []

    def nearby_station(self, room, position):
        matches = [station for station in self.stations if station['room']==room
            and np.linalg.norm(np.array(station['position'][:2])-position[:2])<=.7
            and abs(station['position'][2]-position[2])<=.35]
        return min(matches,key=lambda s:np.linalg.norm(np.array(s['position'])-position),default=None)

    def observe(self, observation, time):
        if (not observation.get("front_frame_available")
                or observation.get('front_valid_depth_fraction',observation.get('valid_depth_fraction',0.))<=.4
                or (observation.get('frame_age_seconds') or 0.)>.65):
            return
        position = np.array(observation["position_estimate"])
        sector = int(math.floor((observation["yaw_degrees"]+22.5)/45)) % 8
        settled = observation.get('speed',0.)<=.1 and abs(observation.get('yaw_rate_rps',0.))<=.1
        sectors = heading_sectors(observation['yaw_degrees'])
        if settled:
            station = self.nearby_station(observation['room'],position)
            if station is None:
                station = {'room':observation['room'],'position':position.tolist(),'surveyed':set()}
                self.stations.append(station)
                self.stations = self.stations[-64:]
            station['surveyed'].update(sectors)
        for view in self.viewpoints.values():
            if view["room"] != observation["room"]:
                continue
            delta = position-view["position"]
            if np.linalg.norm(delta[:2]) <= .7 and abs(delta[2]) <= .35:
                view["headings"].add(sector)
                view["last_observed"] = time
                if settled:
                    view['surveyed'].update(sectors)
                    lo,hi = self.layout.rooms[view['room']]
                    offset = (np.array(lo)+hi)/2-position
                    inward = math.degrees(math.atan2(offset[1],offset[0]))
                    view['inward'] |= abs((inward-observation['yaw_degrees']+180)%360-180)<=45

    def summary(self, room):
        return {name: {"observed_near_viewpoint": bool(view["headings"]),
                      "observed_heading_degrees": sorted(45*h for h in view["headings"]),
                      'surveyed_heading_sectors':sorted(view['surveyed']),
                      'inward_view_observed':bool(view['inward']),
                      "last_observed": view["last_observed"]}
                for name, view in self.viewpoints.items() if view["room"] == room}

    def describe(self, state, criteria):
        state, criteria = dict(state), dict(criteria)
        summary = self.summary(state["sensors"]["room"])
        state["search_memory"] = summary
        state["unobserved_viewpoints"] = [name for name, view in summary.items() if not view["observed_near_viewpoint"]]
        state['viewpoints_without_inward_survey'] = [name for name,view in summary.items() if not view['inward_view_observed']]
        objective = state.get('objective',{})
        if 'marker' in objective:
            known = objective['marker'] in state.get('observed_markers',{}) or state.get('target_visible',False)
            phase = ('REPORT_READY' if state.get('inspection_ready') else
                     'GO_TO_REQUESTED_ROOM' if state['sensors']['room']!=objective['room'] else
                     'APPROACH_OBSERVED_TARGET' if known else 'SEARCH_REQUESTED_ROOM')
            state['search_phase'] = {'situation':phase,'requested_room':objective['room'],
                                     'current_room':state['sensors']['room'],'target_previously_observed':bool(known)}
        pose = state['sensors'].get('position_estimate')
        station = self.nearby_station(state['sensors']['room'],np.array(pose)) if pose is not None else None
        surveyed = station['surveyed'] if station else set()
        state['local_survey'] = {'anchor':list(station['position']) if station else None,
            'surveyed_heading_sectors':sorted(surveyed),
            'unobserved_heading_sectors':sorted(set(range(0,360,45))-surveyed),
            'scope':'Fresh settled front-camera heading coverage near this pose; occluded space remains unknown.'}
        state["options"] = {name: {**option, **summary.get(name, {})} for name, option in state["options"].items()}
        for name, view in summary.items():
            if name in criteria:
                criteria[name] += (" Camera has NOT yet observed from near this viewpoint." if not view["observed_near_viewpoint"]
                                   else " Previously observed near this viewpoint; headings="+str(view["observed_heading_degrees"])+".")
                criteria[name] += ' Inward survey observed='+str(view['inward_view_observed'])+'. Settled front heading sectors='+str(view['surveyed_heading_sectors'])+'.'
        for name,heading in {'look_east':0.,'look_north':90.,'look_west':180.,'look_south':270.,'look_into_room':None}.items():
            if name not in state['options']:
                continue
            if heading is None:
                if pose is None:continue
                target = state['options'][name]['look_position']
                heading = math.degrees(math.atan2(target[1]-pose[1],target[0]-pose[0]))
            fresh = sorted(heading_sectors(heading)-surveyed)
            state['options'][name]['new_heading_sectors'] = fresh
            if name in criteria:
                criteria[name] += ' New front heading sectors from here='+str(fresh)+'. Does not reveal space behind an occluder.'
        criteria["give_up"] += " Do not select while useful unobserved viewpoints remain with adequate time and sensing; a few unsuccessful views do not establish that navigation is impossible."
        return state, criteria
