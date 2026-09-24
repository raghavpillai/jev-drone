"""Jev chooses persistent search/inspection tasks and every camera/flight action."""
from jev_drone.evidence import snapshot_sources
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random

from experiments.legacy.experiment_world import make_scene
from jev_drone.gateway import JevGateway, load_credential
from experiments.legacy.jev_navigation import LOOK
from experiments.legacy.scanning_experiment import ScanningNavigation
from experiments.legacy.search_attention import SearchAttention, visible_probes
from experiments.legacy.search_world import ROOMS, evaluate_report, ground_truth, room_at, update_memory, visible_objects
from experiments.legacy.sim import Simulation, norm, subtract

PLANNING = """Search for the requested item using only observed evidence and the known room layout.
You are responsible for the search strategy. Choose one persistent task for the local Jev pilot.
On entering a room, look around to get an overview. Inspect plausible objects, then move to new views to reveal unseen areas or behind occluders. Look low and high when useful. A camera turn from one position cannot see the entire room. Remember previous attempts; don't repeat a task that revealed nothing unless something changed.
The requested room matters. Transit through known doorways to reach it; align with the near doorway approach before crossing. Rooms are arranged along x. Movement goals are candidate viewpoints, NOT certified free paths; local Jev must avoid obstacles.
inspect means look toward an observed object to clarify its appearance; approach means move to an observed standoff position, stop and look at that object. Report found only if a clearly matching object in the correct room is visible NOW, within 2.15 metres and you are stopped. An ambiguous wooden unit is not evidence of books. Use the requested item's actual description, not just its height or material.
Use view history and candidate visit counts to avoid repeating completed scans/approaches. New-view potential, if present, is an upper bound computed without obstacle knowledge; it may be blocked. Low potential is not proof an item is absent. Keep searching when useful views or uninspected objects remain. Search incomplete is ONLY justified after several failed attempts to reach different views, or after the time allowance is almost exhausted. Not seeing the item yet, or seeing unrelated furniture, is a reason to continue looking. Never report found from memory alone.
"""


class SearchPilot(ScanningNavigation):
    def questions(self):
        questions = super().questions()
        questions['movement']['instructions'] += "\nComplete assigned_task. If requested_look is set and you are inside the arrival radius, select look_REQUESTED_DIRECTION, then brake while viewing it. This inspection is part of arrival: do not simply brake facing away. During travel look toward each needed axis before translating. Do not abandon the assigned waypoint to search independently."
        return questions


def matching_targets(home, item, room):
    # Evaluator-only labels: these IDs must never enter a model request.
    return {o.id for o in home.objects if o.kind == item and
            (room == 'anywhere' or room_at(o.box.lo) == room)}


def mission_scene(seed, case, item, room):
    scene = make_scene(seed, case)
    target = next(iter(sorted(matching_targets(scene.home, item, room))), None)
    home = replace(scene.home, target_id=target)
    scope = tuple(ROOMS) if room == 'anywhere' else (room,)
    return replace(scene, home=home, scope=scope,
                   mission=f'Find a {item} {"anywhere in the home" if room == "anywhere" else "in the " + room}. Approach, stop nearby and visually confirm it. If unable to finish, report search incomplete.')


def look_toward(point, position):
    delta = subtract(point, position)
    return max(LOOK, key=lambda name: sum(a*b for a,b in zip(delta, LOOK[name])))


class ActiveSearch:
    planning_instructions = PLANNING
    api_error_limit = 1

    def __init__(self, args, gateway, scene=None):
        self.args, self.gateway = args, gateway
        self.scene = scene or mission_scene(args.seed, args.case, args.item, args.room)
        self.target_ids = matching_targets(self.scene.home, args.item, args.room)
        self.sim = Simulation(self.scene.home.world)
        self.pilot = SearchPilot(gateway, 'history', True, False, args.seed)
        self.attention = SearchAttention()
        self.memory, self.sightings = {}, Counter()
        self.calls, self.tasks, self.entries = [], [], []
        self.viewpoints = defaultdict(set)
        self.last_room = None
        self.visible, self.task = [], None
        self.error = None
        self.consecutive_api_errors = 0

    def observe(self):
        # Ideal geometry, finite camera cone, synthetic semantic uncertainty.
        raw = visible_objects(self.scene.home, self.sim.position, view_direction=LOOK[self.pilot.look])
        rng = random.Random(self.args.seed*100003 + int(self.sim.time*5)*7 + list(LOOK).index(self.pilot.look))
        self.visible = []
        for obj in raw:
            if rng.random() < .15:
                continue
            self.sightings[obj['id']] += 1
            obj = dict(obj)
            if self.sightings[obj['id']] <= 2 and 'wooden' in obj['appearance']:
                obj['appearance'] = 'A tall wooden storage unit; shelving and contents are unclear from this view.'
            self.visible.append(obj)
        update_memory(self.memory, self.visible, self.sim.position, self.sim.time)
        room = room_at(self.sim.position)
        if room != self.last_room:
            self.entries.append({'room': room, 'time': round(self.sim.time, 2)})
            self.last_room = room
        samples = visible_probes(self.sim.world, self.sim.position, self.pilot.look)
        self.attention.record(samples, self.sim.position, self.pilot.look, self.sim.time, room,
                              (o['id'] for o in self.visible))
        for name, point in self.scene.points.items():
            if norm(subtract(point['xyz'], self.sim.position)) <= .65:
                self.viewpoints[name].add(self.pilot.look)

    def options(self):
        room = room_at(self.sim.position)
        options = {f'look_{look}': {'kind': 'look', 'xyz': list(self.sim.position), 'look': look,
                    'description': f'Stop and look {look} from this position.'} for look in LOOK}
        # Known room topology only. Both sides of a neighboring portal are offered.
        eligible = {name for name, point in self.scene.points.items() if point['room'] == room}
        for i, portal in enumerate(self.scene.portals):
            if room in portal['rooms']:
                eligible.update(f'door_{i}_{r}' for r in portal['rooms'])
        for name in sorted(eligible):
            point = self.scene.points[name]
            entry = {'kind': 'go', **point, 'looked_here': sorted(self.viewpoints[name])}
            if self.args.memory == 'coverage':
                entry['potential_unseen_by_look'] = self.attention.novelty(point['xyz'], point['room'])
            options['go_' + name] = entry
        for obj in self.memory.values():
            if obj['room'] not in self.scene.scope or obj['room'] != room:
                continue
            options['inspect_' + obj['id']] = {'kind': 'inspect', 'xyz': list(self.sim.position),
                'object': obj['id'], 'look': look_toward(obj['visible_surface_xyz'], self.sim.position),
                'description': 'Look toward this remembered surface to clarify/confirm it: ' + obj['appearance']}
            options['approach_' + obj['id']] = {'kind': 'approach', 'xyz': obj['approach_xyz'], 'object': obj['id'],
                'description': 'Move to the observed standoff position, stop and look at: ' + obj['appearance']}
            options['report_found_' + obj['id']] = {'kind': 'report', 'description': 'Report this object as the requested item: ' + obj['appearance']}
        options['report_search_incomplete'] = {'kind': 'report', 'description': 'Unable to finish searching; do not claim absence or success.'}
        return options

    def state(self, options):
        current = room_at(self.sim.position)
        visible_ids = {o['id'] for o in self.visible}
        state = {'mission': self.scene.mission, 'room_bounds': ROOMS, 'doorways': self.scene.portals,
            'time': round(self.sim.time, 2), 'remaining_seconds': round(self.args.seconds-self.sim.time,1),
            'position_xyz': [round(v,2) for v in self.sim.position],
            'speed_mps': round(norm(self.sim.velocity), 2), 'current_room': current,
            'camera_look': self.pilot.look, 'room_entries': list(self.entries),
            'looks_taken_near_this_position': sorted(self.attention.place_looks[tuple(round(v) for v in self.sim.position)]),
            'observed_objects': [{**o, 'visible_now': o['id'] in visible_ids,
                                  'last_seen_ago_s': round(self.sim.time-o['last_seen_at'], 1)} for o in self.memory.values()],
            'recent_task_results': self.tasks[-12:], 'task_counts': dict(Counter(t['choice'] for t in self.tasks)),
            'candidate_tasks': options,
            'sensor_contract': '120 degree camera cone, 3.5m object range. Only observed descriptions/surfaces, never hidden furniture geometry. +x forward, +y left, +z up. Room topology and own pose are known.'}
        if self.args.memory == 'coverage':
            relevant = set(self.scene.scope) | {current}
            state['search_attention'] = {room: self.attention.summary(room) for room in relevant}
            state['recent_views'] = [{'look': v['look'], 'new_visible_samples': v['new_visible_samples'],
                                      'position': v['position']} for v in self.attention.views[-6:]]
            state['potential_unseen_from_here'] = self.attention.novelty(self.sim.position, current)
        return state

    def advance(self, seconds):
        self.sim.advance(min(seconds, max(0., self.args.seconds-self.sim.time)))

    def request_choice(self,role,state,instructions,criteria):
        result = self.gateway.choose(state,instructions,criteria)
        self.calls.append({'role': role, 'sent_at': self.sim.time, **result})
        self.advance(result['latency_seconds'])
        self.handle_api_result(result)
        if result.get('error'):
            return None
        return result['choice']

    def handle_api_result(self,result):
        error=result.get('error')
        if not error:
            self.consecutive_api_errors=0
            return
        self.sim.command('brake')
        self.consecutive_api_errors+=1
        transient=error in {'ReadTimeout','ConnectTimeout','WriteTimeout','PoolTimeout',
                           'ConnectError','ReadError','RemoteProtocolError','http_500','http_502','http_503','http_504'}
        if self.sim.status=='running' and (not transient or self.consecutive_api_errors>=self.api_error_limit):
            self.sim.status,self.error='api_error',error

    def select_task(self,options):
        return self.request_choice('planner',self.state(options),self.planning_instructions,
                                   {name: option['description'] for name, option in options.items()})

    def plan(self):
        self.sim.command('brake')  # Hold for deliberation; no automatic navigation.
        options = self.options()
        choice = self.select_task(options)
        if choice is None or self.sim.status != 'running' or self.sim.time >= self.args.seconds:
            return
        selected = options[choice]
        if selected['kind'] == 'report':
            fresh = visible_objects(self.scene.home, self.sim.position, view_direction=LOOK[self.pilot.look])
            perceived = {o['id'] for o in self.visible if 'unclear from this view' not in o['appearance']}
            if choice == 'report_search_incomplete':
                self.sim.status = 'search_incomplete'
            elif choice.removeprefix('report_found_') not in perceived:
                self.sim.status = 'unverified_report'
            else:
                reported_id = choice.removeprefix('report_found_')
                evaluator_home = replace(self.scene.home, target_id=reported_id if reported_id in self.target_ids else None)
                self.sim.status = evaluate_report(choice, evaluator_home, self.sim, fresh)
            self.tasks.append({'choice': choice, 'outcome': self.sim.status, 'time': self.sim.time})
            return
        self.task = {**selected, 'choice': choice, 'started_at': self.sim.time,
                     'seen_before': len(self.attention.seen), 'objects_before': set(self.memory)}
        self.best = norm(subtract(selected['xyz'], self.sim.position))
        self.progress_at = self.sim.time

    def finish_task(self, outcome):
        event = {'choice': self.task['choice'], 'outcome': outcome, 'time': round(self.sim.time,2),
                 'duration_s': round(self.sim.time-self.task['started_at'],2),
                 'position': [round(v,2) for v in self.sim.position]}
        if self.args.memory == 'coverage':
            event['new_visible_samples'] = len(self.attention.seen)-self.task['seen_before']
        self.tasks.append(event)
        self.task = None

    def task_outcome(self):
        task = self.task
        stopped = norm(self.sim.velocity) <= .15
        arrived = norm(subtract(task['xyz'], self.sim.position)) <= .65 and stopped
        elapsed = self.sim.time-task['started_at']
        if task['kind'] in ('look', 'inspect'):
            if self.pilot.look == task['look'] and stopped and elapsed >= .25:
                return 'inspected' if task['kind'] == 'inspect' else 'looked'
        elif task['kind'] == 'approach':
            if arrived and task['object'] in {o['id'] for o in self.visible}:
                return 'arrived_and_object_visible'
        elif arrived:
            return 'arrived_viewpoint'  # Not equivalent to searched.
        if elapsed >= 24 or self.sim.time-self.progress_at >= 8:
            return 'stalled_choose_another_view_or_route'
        if task['kind'] == 'go' and elapsed > 2 and set(self.memory)-task['objects_before']:
            return 'new_object_observed_reassess'
        return None

    def controller_context(self):
        object_id = self.task.get('object')
        requested = self.task.get('look')
        if self.task['kind'] == 'approach':
            requested = look_toward(self.memory[object_id]['visible_surface_xyz'], self.sim.position)
        return {'task': self.task['kind'], 'requested_look': requested, 'goal': self.task['description']}

    def move(self):
        target = self.task['xyz']
        self.pilot.observed_target = self.memory.get(self.task.get('object'))
        extra = self.controller_context()
        sent = self.sim.time
        result = self.pilot.decide(self.sim, target, extra)
        latency = result['latency_seconds']
        self.advance(latency)
        accepted = self.sim.status == 'running' and self.sim.time < self.args.seconds and latency <= self.sim.command_lease and not result.get('error')
        if self.sim.status == 'running':
            self.sim.command(result['action'] if accepted else 'brake',speed=result.get('speed') if accepted else None)
        self.pilot.accept(self.sim, result, accepted, target)
        self.calls.append({'role': 'control', 'sent_at': sent, 'accepted': accepted, 'task': self.task['choice'], **result})
        self.handle_api_result(result)
        self.advance(max(0., .2-latency))
        distance = norm(subtract(target, self.sim.position))
        if distance < self.best-.15:
            self.best, self.progress_at = distance, self.sim.time

    def run(self):
        while self.sim.status == 'running' and self.sim.time < self.args.seconds-1e-6:
            if sum(c.get('usage',{}).get('cost',0) or 0 for c in self.calls) >= self.args.budget:
                self.sim.status = 'budget_limit'
                break
            self.observe()
            if self.task and (outcome := self.task_outcome()):
                self.finish_task(outcome)
            if self.task is None:
                self.plan()
            elif self.sim.status == 'running':
                self.move()
        if self.sim.status == 'running':
            self.sim.status = 'timeout'
        self.sim.record()
        return {**self.sim.summary(), 'case': self.args.case, 'seed': self.args.seed,
            'item': self.args.item, 'room': self.args.room, 'memory_mode': self.args.memory,
            'mission': self.scene.mission, 'room_entries': self.entries, 'tasks': self.tasks,
            'objects_observed': self.memory, 'coverage': {r:self.attention.summary(r) for r in ROOMS},
            'views': self.attention.views, 'ground_truth': ground_truth(self.scene.home),
            'eligible_target_ids': sorted(self.target_ids),
            'calls': self.calls, 'role_counts': dict(Counter(c['role'] for c in self.calls)),
            'cost_usd': sum(c.get('usage',{}).get('cost',0) or 0 for c in self.calls), 'error': self.error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['near','far','occluded','absent'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--item', default='bookcase')
    parser.add_argument('--room', choices=[*ROOMS,'anywhere'], default='anywhere')
    parser.add_argument('--memory', choices=['checklist','coverage'], default='coverage')
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--budget', type=float, default=.15)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    key = load_credential('openrouter', args.key_file)
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.out / "source", experiments=True)
    (args.out/'protocol.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='key_file'},indent=2))
    gateway = JevGateway('openrouter', key)
    try:
        result = ActiveSearch(args, gateway).run()
    finally:
        gateway.close()
    (args.out/'episode.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('case','seed','item','room','memory_mode','status','simulation_seconds','role_counts','cost_usd','error')}),flush=True)


if __name__ == '__main__':
    main()
