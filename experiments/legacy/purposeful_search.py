"""Experimental Jev-only search: persistent view goals and a single local decision."""
from collections import Counter
from dataclasses import replace

from experiments.legacy.active_search import ActiveSearch, PLANNING, mission_scene
from experiments.legacy.jev_navigation import LOOK, Navigation
from experiments.legacy.search_world import room_at
from experiments.legacy.sim import Box, norm, subtract

PILOT = """Choose the drone's ONE next flight/camera action to complete the assigned task.
Fixed axes: forward +x, back -x, left +y, right -y, up +z, down -z.
A look action brakes and turns the camera; the following request contains that direction's observation. Movement keeps camera aim. Only translate along an axis currently OPEN. UNKNOWN means look in that direction first. Never treat unknown space as free.
Use goal.offset_metres_xyz and the current clearance. Work toward the waypoint one axis at a time. Once you look toward a useful OPEN direction, actually move along it rather than repeatedly switching camera directions.
When inside the arrival radius, stop. If assigned_task.requested_look is set, use its look action to inspect that direction while stopping. Looking is part of the task, including when already at the waypoint.
If an axis toward the goal is blocked, find a different OPEN axis to go around or above it, including moving away temporarily. Use diagonal depth to inspect the space above a barrier. Do not repeatedly move away without checking whether the goal direction has become open. Recent actions/positions show repeated failures; try a different view or detour.
Speed is 1 m/s, acceleration and braking 2 m/s^2; motion continues during network latency. OPEN requires at least 1.3m size-adjusted clearance. Look/brake commands take up to 0.5 seconds to stop.
"""

MOTION_RULES = """
Read flight_status first. If position_reached is true, do NOT translate: choose the required inspection look, or brake. A small remaining coordinate offset is not a reason to translate after arrival.
If stopped is false, do not switch translation directions. First choose brake or a look action and wait until stopped is true, then use a fresh depth observation before moving on a different axis. Drifting on the previous axis while starting a new one produces diagonal motion and can hit a corner even if the new straight-ahead ray is OPEN.
When current_axis_still_needs_travel is false, brake or look toward a remaining goal axis. Do not keep moving just because that direction is OPEN. You have reached or passed that axis's target coordinate; continuing can carry you away from the waypoint or out of the room.
You may continue the SAME current_motion_direction only while that axis still needs travel and the direction is OPEN. If motion is diagonal, brake first. Braking to change axes is normal and necessary; it does not abandon the waypoint. A deliberate obstacle detour can move away temporarily, but must be reassessed rather than continued indefinitely.
"""

CAMERA_RULES = """
Before EVERY translation, the camera must look in that SAME cardinal direction. Camera forward permits only forward translation; to move up, first choose look_up and wait for measured up depth. This applies equally to slow movement. Diagonal rays and previous looks do not make an UNKNOWN cardinal path currently observed.
"""


class DirectPilot(Navigation):
    allow_slow=False

    def decide(self,sim,target,extra=None):
        self.altitude_tolerance=(extra or {}).get('arrival_altitude_tolerance')
        return super().decide(sim,target,extra)

    def observation(self,sim,target):
        common,readings=super().observation(sim,target)
        tolerance=getattr(self,'altitude_tolerance',None)
        if tolerance is not None:
            common['goal']['altitude_within_tolerance']=abs(target[2]-sim.position[2])<=tolerance
            common['goal']['altitude_tolerance_m']=tolerance
            common['goal']['inside_arrival_radius'] &= common['goal']['altitude_within_tolerance']
            if not common['goal']['altitude_within_tolerance']:
                direction='up' if target[2]>sim.position[2] else 'down'
                labels=[v for v in common['goal']['direction'].split('+') if v!='here']
                if direction not in labels:labels.append(direction)
                common['goal']['direction']='+'.join(labels)
        speed=norm(sim.velocity)
        dominant=max(LOOK,key=lambda name:sum(a*b for a,b in zip(sim.velocity,LOOK[name])))
        along=sum(a*b for a,b in zip(sim.velocity,LOOK[dominant]))
        motion='stopped' if speed<=.15 else (dominant if along/speed>=.98 else 'diagonal')
        goal_delta=subtract(target,sim.position)
        remaining_along=sum(a*b for a,b in zip(goal_delta,LOOK[dominant]))
        common['flight_status']={'stopped':speed<=.15,
            'speed_mps':round(speed,2),'slow_speed_ready':speed<=.35,
            'current_motion_direction':motion,
            'position_reached':common['goal']['inside_arrival_radius'],
            'current_axis_still_needs_travel':remaining_along>.35 if motion not in ('stopped','diagonal') else None,
            'remaining_along_current_motion_m':round(remaining_along,2) if motion not in ('stopped','diagonal') else None,
            'stopping_time_seconds':round(norm(sim.velocity)/sim.acceleration,2)}
        return common,readings

    def questions(self):
        questions={'movement': {'type':'choice','instructions':PILOT+MOTION_RULES,'criteria':{
            **{name:f'Move {name} toward an unresolved goal coordinate or for a deliberate short obstacle detour. Requires position_reached false and {name} depth OPEN. If still moving, requires that you are already moving {name} AND current_axis_still_needs_travel is true; otherwise brake/look first.' for name in LOOK},
            **{'look_'+name:f'Brake and aim camera {name}; prepare to move along a remaining goal axis or inspect this direction on the next request.' for name in LOOK},
            'brake':'Stop to change movement axes, when the current axis has reached/passed its goal coordinate, when arrived, or for safety. Do not continue moving away just because the current direction is open.'}}}
        if self.allow_slow:
            questions['movement']['instructions']+='\nSlow actions move at 0.3 m/s. Use them for small remaining offsets, altitude precision, and clearances too short for cruising. A BLOCKED label means blocked for FAST 1m/s travel: a SLOW action is possible if its numeric clearance is at least 0.45m. Never use an UNKNOWN reading. Brake first if still moving fast, or before changing axes. When altitude_within_tolerance is false, finish the altitude adjustment before treating the viewpoint as reached.'
            questions['movement']['criteria'].update({'slow_'+name:f'Move slowly {name} at 0.3m/s for a small correction or short clearance. Requires measured {name} clearance >=0.45m, slow_speed_ready true, waypoint not reached, and stopped or already moving {name}. UNKNOWN or <0.45m forbids this.' for name in LOOK})
        questions['movement']['instructions']+=CAMERA_RULES
        for name in LOOK:
            for action in (name,'slow_'+name):
                if action in questions['movement']['criteria']:
                    questions['movement']['criteria'][action]+=f' The camera MUST currently look {name}; if it looks elsewhere, choose look_{name} first.'
        return questions

    def decode(self, result):
        if result.get('error'):
            result['action']='brake'
            return result
        selected=result['answers']['movement']['choice']
        result.update(selected_action=selected,action='brake' if selected.startswith('look_') else selected.removeprefix('slow_'),
                      speed=.3 if selected.startswith('slow_') else 1.,
                      look=selected.removeprefix('look_') if selected.startswith('look_') else self.look,intent='direct')
        return result

    def accept(self,sim,decision,accepted,target):
        super().accept(sim,decision,accepted,target)
        self.history[-1]['camera_look']=self.look
        self.history[-1]['selected_action']=decision.get('selected_action','brake') if accepted else 'brake'


def varied_scene(args):
    scene=mission_scene(args.seed,args.case,args.item,args.room)
    variation=getattr(args,'variation','original')
    objects=[]
    for obj in scene.home.objects:
        lo,hi=list(obj.box.lo),list(obj.box.hi)
        offset=int(lo[0]//6)*6
        if variation in ('mirror_x','mirror_xy'):
            lo[0],hi[0]=2*offset+6-hi[0],2*offset+6-lo[0]
        if variation in ('mirror_y','mirror_xy'):
            lo[1],hi[1]=10-hi[1],10-lo[1]
        if variation in ('east_wall','west_wall') and obj.kind in ('bookcase','screen'):
            if variation=='east_wall':
                lo[:2],hi[:2]=([offset+5.1,8.5],[offset+5.7,9.9]) if obj.kind=='bookcase' else ([offset+4.1,6.9],[offset+4.5,9.9])
            else:
                lo[:2],hi[:2]=([offset+.3,.3],[offset+.9,1.7]) if obj.kind=='bookcase' else ([offset+1.4,.1],[offset+1.6,3.1])
        if obj.kind=='screen':
            hi[2]=getattr(args,'screen_height',hi[2])
        objects.append(replace(obj,box=Box(tuple(lo),tuple(hi))))
    # The first four solids are known room walls; all later ones are hidden objects.
    world=replace(scene.home.world,obstacles=scene.home.world.obstacles[:4]+tuple(o.box for o in objects))
    return replace(scene,home=replace(scene.home,world=world,objects=tuple(objects)))


class PurposefulSearch(ActiveSearch):
    planning_instructions=PLANNING+"""
Viewpoint tasks now include a viewing direction: travel there, stop, and inspect that view before reconsidering. High viewpoints are useful for seeing over a low obstacle. Their viewing direction matters; climbing then immediately looking away can miss the hidden object.
The failed_attempts attached to each option summarize your actual previous attempts. A repeatedly stalled goal needs a different intermediate position or height before retrying. A new camera direction from the same stuck position does not change the line of sight. Stay in the requested room while searching it. Look for objects on any side of the room, not just one wall.
"""

    def __init__(self,args,gateway):
        super().__init__(args,gateway,varied_scene(args))
        if getattr(args,'pilot','direct')=='direct':
            self.pilot=DirectPilot(gateway,'history',True,False,args.seed)
        self.view_completions=Counter()

    def options(self):
        options=super().options()
        for name,option in list(options.items()):
            if not name.startswith('go_search_'):
                continue
            direction='right' if 'south_' in name else 'left'
            option['look']=direction
            option['description']+=f' Stop and inspect {direction} after arrival.'
            if not name.endswith('_high'):
                high={**option,'xyz':[option['xyz'][0],option['xyz'][1],2.8],
                      'description':option['description']+' Use a high view at 2.8m to look over low occluders.'}
                options[name+'_high']=high
            else:
                options[name+'_south']={**option,'look':'right','description':
                    f'Look across the south half of {option["room"]} from 2.8m; stop and inspect right after arrival.'}
        for name,option in options.items():
            if self.args.memory=='coverage' and option['kind']=='go':
                option['potential_unseen_by_look']=self.attention.novelty(option['xyz'],option['room'])
            attempts=[t for t in self.tasks if t['choice']==name]
            option['completed_views']=self.view_completions[name]
            option['failed_attempts']=sum(t['outcome'].startswith('stalled') for t in attempts)
            if attempts:
                option['last_outcome']=attempts[-1]['outcome']
        return options

    def controller_context(self):
        context=super().controller_context()
        tolerance=self.task.get('altitude_tolerance')
        if not self.at_task_position():
            context={'phase':'travel','task':'reach_waypoint','requested_look':None,
                    'goal':'Reach the current waypoint and stop. Inspecting the final view is a separate later phase.'}
        else:
            context={**context,'phase':'inspect_and_stop'}
        return {**context,'arrival_altitude_tolerance':tolerance}

    def at_task_position(self):
        return (norm(subtract(self.task['xyz'],self.sim.position))<=.65 and
                abs(self.task['xyz'][2]-self.sim.position[2])<=self.task.get('altitude_tolerance',float('inf')))

    def task_outcome(self):
        task=self.task
        if task['kind']=='go' and task.get('look'):
            arrived=self.at_task_position() and norm(self.sim.velocity)<=.15
            if arrived and self.pilot.look==task['look'] and self.sim.time-task['started_at']>=.25:
                return 'viewpoint_inspected'
            # Arrival is not completion when the inspection direction is still missing.
            if self.sim.time-task['started_at']>=24 or self.sim.time-self.progress_at>=8:
                return 'stalled_choose_another_view_or_route'
            if self.sim.time-task['started_at']>2 and set(self.memory)-task['objects_before']:
                return 'new_object_observed_reassess'
            return None
        return super().task_outcome()

    def finish_task(self,outcome):
        if outcome=='viewpoint_inspected':
            self.view_completions[self.task['choice']]+=1
        super().finish_task(outcome)

    def run(self):
        result=super().run()
        result.update(experiment='purposeful-v4',variation=getattr(self.args,'variation','original'),
                      pilot=getattr(self.args,'pilot','direct'),completed_view_tasks=dict(self.view_completions))
        return result
