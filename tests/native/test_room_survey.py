from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from jev_drone.sim.config import Config
from jev_drone.control.policy import FrontExperiment
from jev_drone.planning.search_memory import SearchMemory
from jev_drone.planning.room_intent import describe as room_intent
from jev_drone.world.layout import HOUSE, SMALL


def observation(**changes):
    return dict(room='workshop',position_estimate=[19.5,11.,1.3],yaw_degrees=90.,
        speed=0.,yaw_rate_rps=0.,front_frame_available=True,front_valid_depth_fraction=.9,
        valid_depth_fraction=.9,frame_age_seconds=.2,detections=[],**changes)


def test_flybys_do_not_count_as_settled_surveys_or_inward_views():
    memory=SearchMemory(HOUSE)
    obs=observation();obs['speed']=.25
    memory.observe(obs,1.)
    view=memory.summary('workshop')['view_workshop_0_1_1.3']
    assert view['observed_near_viewpoint']
    assert not view['surveyed_heading_sectors'] and not view['inward_view_observed']
    obs['speed']=0.
    memory.observe(obs,2.)
    assert not memory.summary('workshop')['view_workshop_0_1_1.3']['inward_view_observed']
    obs['yaw_degrees']=-53.
    memory.observe(obs,3.)
    assert memory.summary('workshop')['view_workshop_0_1_1.3']['inward_view_observed']


def test_front_quality_and_freshness_cannot_be_replaced_by_auxiliary_depth():
    for change in ({'front_valid_depth_fraction':0.},{'front_frame_available':False},{'frame_age_seconds':.8}):
        memory=SearchMemory(HOUSE);obs={**observation(),**change};memory.observe(obs,1.)
        assert not memory.stations
        assert not any(v['observed_near_viewpoint'] for v in memory.summary('workshop').values())


def test_local_heading_coverage_does_not_follow_the_drone_to_other_positions_or_heights():
    memory=SearchMemory(HOUSE);obs=observation()
    for heading in (0.,90.,180.,270.):
        memory.observe({**obs,'yaw_degrees':heading},heading+1)
    assert memory.nearby_station('workshop',np.array(obs['position_estimate']))['surveyed']==set(range(0,360,45))
    assert memory.nearby_station('workshop',np.array([21.,11.,1.3])) is None
    assert memory.nearby_station('workshop',np.array([19.5,11.,2.2])) is None
    assert not memory.summary('workshop')['view_workshop_0_1_2.2']['surveyed_heading_sectors']


def test_phase_and_look_novelty_are_facts_not_automatic_choices():
    memory=SearchMemory(HOUSE);obs=observation();memory.observe(obs,1.)
    state={'objective':{'marker':'red','room':'workshop'},'sensors':obs,
        'options':{'look_north':{'look':'north'},'look_south':{'look':'south'},'give_up':{}}}
    original=deepcopy(state);criteria=dict.fromkeys(state['options'],'Original.')
    updated,choices=memory.describe(state,criteria)
    assert state==original and choices.keys()==criteria.keys()
    assert updated['search_phase']['situation']=='SEARCH_REQUESTED_ROOM'
    assert updated['options']['look_north']['new_heading_sectors']==[]
    assert updated['options']['look_south']['new_heading_sectors']==[225,270,315]
    outside={**state,'sensors':{**obs,'room':'library'}}
    assert memory.describe(outside,criteria)[0]['search_phase']['situation']=='GO_TO_REQUESTED_ROOM'
    lead={**state,'observed_markers':{'red':{'position':[22.,10.,1.6]}}}
    assert memory.describe(lead,criteria)[0]['search_phase']['situation']=='APPROACH_OBSERVED_TARGET'
    assert memory.describe({**lead,'inspection_ready':True},criteria)[0]['search_phase']['situation']=='REPORT_READY'


def experiment():
    experiment=object.__new__(FrontExperiment)
    experiment.layout=HOUSE;experiment.config=Config();experiment.stage=0
    experiment.world=SimpleNamespace(mission=[{'marker':'red','room':'workshop'}])
    experiment.memory={};experiment.tasks=[];experiment.search_memory=SearchMemory(HOUSE)
    experiment.sensors=SimpleNamespace(observed_gap_at=lambda p:None,current=[])
    return experiment


def test_current_house_view_tasks_face_inward_and_share_feasible_heights_with_memory():
    agent=experiment()
    choices=agent.options(observation())
    for name,option in choices.items():
        if name.startswith('view_'):
            assert option['position']==agent.search_memory.viewpoints[name]['position']
            assert option['look_position'][:2]==[21.,9.]
            assert option['position'][2] in (1.3,2.2)
    assert 'look_into_room' in choices
    assert {p['position'][2] for p in SMALL.search_viewpoints().values()}=={1.3,2.9}


def test_completed_stationary_look_cannot_form_an_immediate_replanning_loop():
    agent=experiment()
    obs={**observation(),'position_estimate':[22.23,8.96,2.06],'yaw_degrees':179.}
    choices=agent.options(obs)
    assert 'look_into_room' not in choices and 'look_west' not in choices
    assert {'look_north','look_south','look_east'} <= choices.keys()
    assert len([name for name in choices if name.startswith('view_')])==16
    assert 'look_into_room' in agent.options({**obs,'yaw_degrees':90.})
    assert 'look_west' not in agent.options({**obs,'yaw_degrees':-179.})


def test_look_is_retained_until_settled_with_fresh_front_evidence():
    agent=experiment()
    obs={**observation(),'position_estimate':[22.23,8.96,2.06],'yaw_degrees':179.}
    for change in ({'speed':.2},{'yaw_rate_rps':.2},{'front_frame_available':False},
                   {'front_valid_depth_fraction':0.},{'frame_age_seconds':.8},{'frame_age_seconds':None}):
        assert 'look_into_room' in agent.options({**obs,**change})


def test_outbound_room_intent_does_not_leak_into_return_or_next_item():
    state={'stage':1,'sensors':{'room':'library'},'recent_tasks':[
        {'name':'cross_library_workshop','mission_stage':0,'leads_to_room':'workshop','outcome':'arrived'}],
        'options':{'approach_library_workshop':{},'approach_bedroom_library':{}}}
    updated,_=room_intent(state,dict.fromkeys(state['options'],'Original.'))
    assert updated['room_transit_intent'] is None
