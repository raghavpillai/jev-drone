from types import SimpleNamespace

from experiments.legacy.purposeful_search import DirectPilot, PurposefulSearch, varied_scene
from experiments.legacy.sim import norm, subtract


def config(**changes):
    return SimpleNamespace(**(dict(case='occluded',seed=502,item='bookcase',room='bedroom',memory='checklist',
        seconds=150,budget=.2,pilot='direct',variation='original')|changes))


def test_local_action_has_no_stale_maneuver_latch():
    pilot=DirectPilot(None,limited=True)
    pilot.intent='retreat'
    assert set(pilot.questions())=={'movement'}
    result=pilot.decode({'answers':{'movement':{'choice':'forward'}}})
    assert result['action']=='forward' and result['intent']=='direct'


def test_viewpoint_requires_look_after_arrival():
    experiment=PurposefulSearch(config(),None)
    experiment.sim.time=1
    experiment.progress_at=0
    experiment.task={'kind':'go','xyz':list(experiment.sim.position),'look':'left','started_at':0,'objects_before':set()}
    experiment.pilot.look='forward'
    assert experiment.task_outcome() is None
    experiment.pilot.look='left'
    assert experiment.task_outcome()=='viewpoint_inspected'


def test_high_views_are_room_derived_without_hidden_furniture():
    original=PurposefulSearch(config(),None)
    reflected=PurposefulSearch(config(variation='mirror_xy'),None)
    assert original.options()==reflected.options()
    options=original.options()
    high=options['go_search_living_room_north_east_high']
    assert high['xyz'][2]==2.8 and high['look']=='left'
    assert 'go_search_living_room_south_east_high' in options


def test_scene_variation_moves_target_and_partition_but_not_known_doors():
    original=varied_scene(config())
    reflected=varied_scene(config(variation='mirror_y'))
    assert original.portals==reflected.portals and original.points==reflected.points
    assert original.home.world.obstacles[:4]==reflected.home.world.obstacles[:4]
    for a,b in zip(original.home.objects,reflected.home.objects):
        assert a.id==b.id and a.kind==b.kind
        assert abs(a.box.hi[1]+b.box.lo[1]-10)<1e-6
    target=next(o for o in reflected.home.objects if o.kind=='bookcase')
    assert target.box.hi[1]<1


def test_failed_task_metadata_is_preserved_without_masking_choice():
    experiment=PurposefulSearch(config(),None)
    choice='go_search_living_room_north_east'
    experiment.tasks=[{'choice':choice,'outcome':'stalled_choose_another_view_or_route'}]*2
    options=experiment.options()
    assert options[choice]['failed_attempts']==2
    assert choice in options


def test_inspection_direction_is_withheld_until_arrival():
    experiment=PurposefulSearch(config(),None)
    experiment.task={'kind':'go','xyz':[9,6,2.8],'look':'left','description':'Inspect left at the high viewpoint.'}
    context=experiment.controller_context()
    assert context['phase']=='travel' and context['requested_look'] is None
    experiment.sim.position=(9,6,2.8)
    context=experiment.controller_context()
    assert context['phase']=='inspect_and_stop' and context['requested_look']=='left'


def test_controller_remembers_actual_look_actions():
    experiment=PurposefulSearch(config(),None)
    decision=experiment.pilot.decode({'answers':{'movement':{'choice':'look_up'}}})
    experiment.pilot.accept(experiment.sim,decision,True,(9,6,2.8))
    assert experiment.pilot.history[-1]['selected_action']=='look_up'
    assert experiment.pilot.history[-1]['camera_look']=='up'


def test_side_wall_scenes_reposition_occluder_and_keep_room_walls():
    original=varied_scene(config())
    for variation in ('east_wall','west_wall'):
        scene=varied_scene(config(variation=variation,screen_height=2.45))
        assert scene.home.world.obstacles[:4]==original.home.world.obstacles[:4]
        bookcase=next(o for o in scene.home.objects if o.kind=='bookcase')
        screen=next(o for o in scene.home.objects if o.kind=='screen')
        assert abs(bookcase.box.hi[0]-bookcase.box.lo[0]-.6)<1e-6
        assert screen.box.hi[2]==2.45
        for obj in scene.home.objects:
            assert all(0<=a<b<=limit for a,b,limit in zip(obj.box.lo,obj.box.hi,(18,10,4)))


def test_motion_facts_expose_drift_without_masking_model_actions():
    experiment=PurposefulSearch(config(),None)
    experiment.sim.velocity=(.14,.14,0)
    state,_=experiment.pilot.observation(experiment.sim,(9,6,2.8))
    assert state['flight_status']['current_motion_direction']=='diagonal'
    assert not state['flight_status']['stopped']
    experiment.sim.velocity=(0,1,0)
    state,_=experiment.pilot.observation(experiment.sim,experiment.sim.position)
    assert state['flight_status']['current_motion_direction']=='left'
    assert state['flight_status']['position_reached']
    # A model error is still executed and scored; there is no hidden action veto.
    decoded=experiment.pilot.decode({'answers':{'movement':{'choice':'back'}}})
    assert decoded['action']=='back'


def test_motion_progress_fact_detects_passing_an_axis_target():
    experiment=PurposefulSearch(config(),None)
    experiment.sim.position=(9.5,5,1.4)
    experiment.sim.velocity=(1,0,0)
    state,_=experiment.pilot.observation(experiment.sim,(9,7,2.8))
    motion=state['flight_status']
    assert motion['current_motion_direction']=='forward'
    assert not motion['current_axis_still_needs_travel']
    assert motion['remaining_along_current_motion_m']==-.5
    assert not motion['position_reached']
