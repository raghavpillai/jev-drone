"""Synthetic all-Jev room-choice probe; graph distances are evaluator-only."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.agent import PLANNER
from jev_drone.world.layout import HOUSE
from jev_drone.sim.config import Config
from jev_drone.control.policy import FrontExperiment
from jev_drone.planning.search_memory import SearchMemory
from jev_drone.planning.return_memory import ReturnMemory
from jev_drone.planning.door_memory import DoorMemory


def distance(start, destination):
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        room, depth = queue.popleft()
        if room == destination:
            return depth
        for edge in HOUSE.neighbors()[room]:
            if edge['room'] not in seen:
                seen.add(edge['room'])
                queue.append((edge['room'], depth+1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gateway = JevGateway('openrouter', load_credential('openrouter', Path('.env')),
                         journal=args.out/'calls.jsonl')
    rows = []
    try:
        for destination in ('workshop', 'entry'):
            for room, (lo, hi) in HOUSE.rooms.items():
                if room == destination:
                    continue
                for repeat in range(2):
                    experiment = object.__new__(FrontExperiment)
                    experiment.layout, experiment.config, experiment.gateway = HOUSE, Config(policy='v4'), gateway
                    objective = {'marker':'red', 'room':destination}
                    experiment.world = SimpleNamespace(time=0., realtime=True, mission=[objective])
                    experiment.sensors = SimpleNamespace(current=[], observed_gap_at=lambda p: None)
                    experiment.stage, experiment.calls, experiment.memory, experiment.tasks = 0, [], {}, []
                    experiment.cost, experiment.error_streak = 0., 0
                    experiment.visits = Counter()
                    experiment.front_views = []
                    experiment.search_memory, experiment.return_memory, experiment.door_memory = SearchMemory(HOUSE), ReturnMemory(), DoorMemory(HOUSE)
                    pose = ((np.array(lo)+hi)/2).tolist(); pose[2] = 1.3
                    observation = dict(room=room, position_estimate=pose, detections=[], speed=0., yaw_rate_rps=0.)
                    options = experiment.options(observation)
                    state = dict(mission=[objective], stage=0, objective=objective, ACTIVE_OBJECTIVE=objective,
                        inspection_ready=False, dock_ready=False, target_visible=False, time_remaining=900,
                        sensors=observation, known_room_bounds=HOUSE.rooms, known_doors=HOUSE.doors,
                        known_room_connections=HOUSE.neighbors(), options=options,
                        observed_markers={}, recent_tasks=[], visits={})
                    choice, log = experiment.request('planner', state, PLANNER, {k:json.dumps(v) for k,v in options.items()})
                    next_room = options.get(choice,{}).get('leads_to_room')
                    row = dict(room=room, destination=destination, repeat=repeat, choice=choice, next_room=next_room,
                        reduces_hops=next_room is not None and distance(next_room,destination)<distance(room,destination),
                        error=log.get('error'), cost_usd=experiment.cost)
                    rows.append(row);print(json.dumps(row),flush=True)
    finally:
        gateway.close()
        (args.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.out/'protocol.json').write_text(json.dumps({'kind':'synthetic room reasoning, not a flight',
            'states':14,'repetitions':2,'expected_requests':56,
            'score':'Does the selected next room reduce unblocked architectural graph distance? Distances never enter model input.'},indent=2))


if __name__ == '__main__':
    main()
