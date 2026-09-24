"""Geometry and information boundaries for missions spanning many rooms."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from jev_drone.world.layout import HOUSE, SMALL, HOUSE_CASES, Layout
from jev_drone.sim.scene import Geometry, generate
from jev_drone.sim.config import Config
from jev_drone.control.doorway_control import heading, describe
from jev_drone.planning.door_memory import DoorMemory
from jev_drone.planning.detour_route import candidate
from jev_drone.control.policy import FrontExperiment
from jev_drone.planning.search_memory import SearchMemory


def hops(layout, start, destination):
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        room, distance = queue.popleft()
        if room == destination:
            return distance
        for edge in layout.neighbors()[room]:
            if edge['room'] not in seen:
                seen.add(edge['room'])
                queue.append((edge['room'], distance+1))


@pytest.mark.parametrize('case', [case for case in HOUSE_CASES if not case.startswith('house_search')])
def test_house_has_distant_reachable_objectives_and_clear_door_centers(case):
    scene = Geometry(Config(case=case, seed=5101))
    assert len(scene.layout.rooms) == 8
    assert hops(HOUSE, HOUSE.room_at(scene.start), scene.mission[0]['room']) == 4
    assert np.linalg.norm(scene.start-scene.targets['red']['position']) > 20
    for name, center in HOUSE.doors.items():
        gaps = [np.linalg.norm(np.maximum(np.abs(np.array(center)-b['center'])-b['half'], 0))
                for b in scene.geometry if b['name'] != 'unexpected_closed_door']
        assert min(gaps) >= .8-1e-6, name
    for objective in scene.mission[:-1]:
        assert HOUSE.room_at(scene.targets[objective['marker']]['position']) == objective['room']
    assert scene.mission[-1]['dock'] == list(scene.start)


def test_search_diagnostics_share_hidden_geometry_without_disclosing_target_presence():
    present = Geometry(Config(case='house_search', seed=5111))
    absent = Geometry(Config(case='house_search_absent', seed=5111))
    full = Geometry(Config(case='house_hidden', seed=5111))
    assert present.mission == absent.mission
    assert HOUSE.room_at(present.start) == 'workshop'
    assert present.targets == full.targets
    assert 'red' not in absent.targets
    geometry = lambda scene: [b for b in scene.geometry if not b['name'].startswith('red_')]
    # Body identifiers shift when a marker is removed; compare physical parts.
    physical = lambda scene: [{k:v for k,v in b.items() if k != 'body'} for b in geometry(scene)]
    assert physical(present) == physical(absent) == physical(full)


@pytest.mark.parametrize('case', ['house_hidden', 'house_reverse'])
def test_find_only_keeps_scene_and_inspection_objective(case):
    full = Geometry(Config(case=case, seed=5811))
    search = Geometry(Config(case=case, seed=5811, find_only=True))
    assert search.mission == full.mission[:1]
    assert search.geometry == full.geometry
    assert search.targets == full.targets
    assert np.array_equal(search.start, full.start)


def test_room_names_do_not_encode_axis_direction_or_launch_room():
    layout = Layout({'north_room': ((0,6,0),(6,12,4)), 'south_room': ((0,0,0),(6,6,4))},
                    {'portal_1': (3,6,1.3)}, {'portal_1': ('north_room','south_room')})
    experiment = object.__new__(FrontExperiment)
    experiment.layout = layout
    experiment.config = Config()
    experiment.world = SimpleNamespace(mission=[{'dock':[1.5,9,1.3]}])
    experiment.stage, experiment.memory, experiment.tasks = 0, {}, []
    experiment.search_memory = SearchMemory(layout)
    experiment.return_memory = SimpleNamespace(options=lambda *args: {})
    experiment.sensors = SimpleNamespace(observed_gap_at=lambda point: None, current=[])
    obs = dict(room='south_room', position_estimate=[3,5,1.3], detections=[], speed=0., yaw_rate_rps=0.)
    options = experiment.options(obs)
    task = {'name':'cross_portal_1', **options['cross_portal_1']}
    assert task['position'] == [3,7,1.3] and task['leads_to_room'] == 'north_room'
    assert heading(task, layout) == 90.
    experiment.task = task
    assert experiment.crossed_doorway({**obs,'room':'north_room','position_estimate':[3,6.7,1.3]})
    north = experiment.options({**obs,'room':'north_room','position_estimate':[3,7,1.3]})
    assert north['cross_portal_1']['position'] == [3,5,1.3]
    assert north['cross_portal_1']['heading_degrees'] == -90.
    assert 'return_launch' in north and 'return_launch' not in options


def test_observed_obstruction_uses_layout_specific_door_axis():
    # living_bedroom was east/west in the old scene; here it is north/south.
    task = {'name':'cross_living_bedroom', 'position':[9,7,1.3]}
    assert heading(task, HOUSE) == 90.
    assert heading({'name':'cross_living_bedroom','position':[7,2,1.3]}, SMALL) == 0.
    cloud = np.array([[9+a,5.92,z] for a in np.linspace(-.45,.45,10) for z in np.linspace(.85,2,10)])
    memory = DoorMemory(HOUSE)
    memory.observe(cloud, 2.)
    assert set(memory.obstructions) == {'living_bedroom'}
    state, _ = describe({'task':task,'sensors':{'position_estimate':[9.2,5,1.3],'yaw_degrees':90}}, {}, HOUSE)
    assert state['doorway_alignment']['centerline_offset_m'] == .2


def test_detour_candidates_and_search_memory_extend_beyond_old_house():
    route = candidate([20,9,1.3], [22,9,1.3], [0,1,0], .9, True, HOUSE)
    assert np.allclose(route, [[20,9.9,1.3],[22,9.9,1.3]])
    memory = SearchMemory(HOUSE)
    assert len(memory.viewpoints) == 128
    assert len(memory.summary('workshop')) == 16


def test_edge_candidates_offer_an_alternative_to_a_shelf_blocked_corner():
    scene = Geometry(Config(case='house_hidden', seed=5111))
    views = scene.layout.search_viewpoints()
    def evaluator_gap(position):
        return min(np.linalg.norm(np.maximum(np.abs(np.array(position)-box['center'])-box['half'],0))-.4
                   for box in scene.geometry)
    for height in (1.3,2.2):
        assert evaluator_gap(views[f'view_workshop_1_1_{height}']['position']) < 0.
        assert evaluator_gap(views[f'view_workshop_edge_east_{height}']['position']) > .08
    # Offering a candidate does not certify it clear or place it at a target.
    assert all(set(option)=={'room','position'} for option in views.values())


def test_reverse_launch_sdf_and_known_layout_match_metadata(tmp_path):
    source = Path('.sim/PX4-Autopilot')
    if not source.exists():
        pytest.skip('Pinned simulator source required')
    metadata = generate(Config(case='house_reverse'), source, tmp_path)
    drone = ET.parse(tmp_path/'world.sdf').find("world/model[@name='drone']")
    assert list(map(float,drone.findtext('pose').split()))[:3] == [22.5,9.,.24]
    assert len(metadata['rooms']) == 8 and len(metadata['room_connections']) == 10
    assert all('target' not in key and 'furniture' not in key for key in HOUSE.neighbors())
