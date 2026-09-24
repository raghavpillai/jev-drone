"""Evaluator-only geometric feasibility; no routes or truth enter the flight policy.

Flood a horizontal grid with a conservative spherical body. This checks that
furniture does not make inspection impossible, not that the sensor-limited
controller can discover a route or fly it with real dynamics.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np

from jev_drone.sim.config import Config
from jev_drone.sim.scene import Geometry
from jev_drone.world.layout import HOUSE_CASES


def visible(origin, target, geometry, target_body):
    boxes = [box for box in geometry if box['body'] != target_body]
    lower = np.array([b['center'] for b in boxes])-np.array([b['half'] for b in boxes])
    upper = np.array([b['center'] for b in boxes])+np.array([b['half'] for b in boxes])
    enter, leave = np.zeros(len(boxes)), np.ones(len(boxes))
    for axis, delta in enumerate(target-origin):
        if abs(delta) < 1e-9:
            leave[(origin[axis] < lower[:,axis]) | (origin[axis] > upper[:,axis])] = -1
        else:
            a, b = (lower[:,axis]-origin[axis])/delta, (upper[:,axis]-origin[axis])/delta
            enter = np.maximum(enter,np.minimum(a,b))
            leave = np.minimum(leave,np.maximum(a,b))
    return not np.any(enter <= leave)


def check(case):
    scene = Geometry(Config(case=case, seed=5111))
    step, radius, altitude = .25, .45, 1.3
    lo, hi = scene.layout.bounds
    xs = np.arange(lo[0],hi[0]+step/2,step)
    ys = np.arange(lo[1],hi[1]+step/2,step)
    xx, yy = np.meshgrid(xs,ys,indexing='ij')
    points = np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,altitude)))
    gaps = np.full(len(points),np.inf)
    for box in scene.geometry:
        gaps = np.minimum(gaps,np.linalg.norm(np.maximum(np.abs(points-box['center'])-box['half'],0),axis=1))
    # A point along a grid edge is at most step/2 from an endpoint, so this
    # margin certifies at least radius clearance along every connected edge.
    free = (gaps >= radius+step/2).reshape(xx.shape)
    start = tuple(np.rint((scene.start[:2]-lo[:2])/step).astype(int))
    assert free[start], f'{case}: launch grid cell is occupied'
    distances, queue = {start:0.}, deque([start])
    while queue:
        cell = queue.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            neighbor = cell[0]+dx,cell[1]+dy
            if (0 <= neighbor[0] < len(xs) and 0 <= neighbor[1] < len(ys)
                    and free[neighbor] and neighbor not in distances):
                distances[neighbor] = distances[cell]+step
                queue.append(neighbor)
    inspections = []
    for objective in scene.mission[:-1]:
        target = scene.targets[objective['marker']]
        position = np.array(target['position'])
        candidates = []
        for cell, distance in distances.items():
            point = np.array([xs[cell[0]],ys[cell[1]],altitude])
            if (scene.layout.room_at(point)==objective['room']
                    and np.linalg.norm(point-position) <= 1.5
                    and visible(point,position,scene.geometry,target['body'])):
                candidates.append(distance)
        assert candidates, f'{case}: no reachable line-of-sight inspection position for {objective}'
        inspections.append({'marker':objective['marker'],'room':objective['room'],
            'reachable_inspection_positions':len(candidates),'grid_distance_from_launch_m':min(candidates)})
    hidden = None
    if case in ('house_hidden','house_sequence'):
        target = scene.targets['red']
        hidden = not visible(np.array([19.,9.,altitude]),np.array(target['position']),scene.geometry,target['body'])
        assert hidden, 'The hidden target is directly visible on entering from the library'
    return {'case':case,'seed':5111,'collision_parts':len(scene.geometry),
        'reachable_grid_cells':len(distances),'grid_spacing_m':step,'body_radius_m':radius,
        'altitude_m':altitude,'inspection_positions':inspections,'hidden_from_library_entry':hidden}


def main():
    rows = [check(case) for case in HOUSE_CASES]
    output = Path('report/px4-house/layout-feasibility.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'kind':'Evaluator truth only; no model calls or native flights',
        'limits':'Static sphere at one altitude; excludes sensing, dynamic braking and actual airframe attitude.',
        'cases':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
