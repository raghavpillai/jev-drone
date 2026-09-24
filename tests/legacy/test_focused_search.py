from experiments.legacy.focused_search import FocusedSearch
from tests.legacy.test_purposeful_search import config


class ScriptedGateway:
    def __init__(self,choices):
        self.choices=iter(choices)
        self.requests=[]

    def choose(self,state,instructions,criteria):
        self.requests.append((state,criteria))
        choice=next(self.choices)
        assert choice in criteria
        return {'choice':choice,'latency_seconds':.1,'usage':{}}


def test_continue_and_task_choices_are_separate_jev_requests():
    gateway=ScriptedGateway(['continue_search','look_left'])
    experiment=FocusedSearch(config(),gateway)
    experiment.observe()
    experiment.plan()
    assert experiment.task['choice']=='look_left'
    stop_criteria=gateway.requests[0][1]
    task_criteria=gateway.requests[1][1]
    assert 'continue_search' in stop_criteria and 'look_left' not in stop_criteria
    assert 'look_left' in task_criteria and not any(k.startswith('report_') for k in task_criteria)
    assert [c['role'] for c in experiment.calls]==['mission','planner']
    assert abs(experiment.sim.time-.2)<1e-6


def test_jev_can_stop_without_a_task_request():
    gateway=ScriptedGateway(['report_search_incomplete'])
    experiment=FocusedSearch(config(),gateway)
    experiment.observe()
    experiment.plan()
    assert experiment.sim.status=='search_incomplete'
    assert len(gateway.requests)==1


def test_interrupted_view_remains_unfinished():
    experiment=FocusedSearch(config(),None)
    goal='go_search_living_room_north_west'
    experiment.tasks=[{'choice':goal,'outcome':'new_object_observed_reassess'}]
    state=experiment.stopping_state(experiment.options())
    assert goal in state['remaining_view_goals']


def test_confirmation_readiness_uses_observed_range_and_velocity():
    experiment=FocusedSearch(config(),None)
    point=[experiment.sim.position[0]+3,experiment.sim.position[1],experiment.sim.position[2]]
    obj={'id':'observed','visible_surface_xyz':point}
    experiment.visible=[obj]
    facts=experiment.confirmation_facts(obj)
    assert facts['visible_now'] and not facts['physical_confirmation_ready']
    point[0]-=1.2
    assert experiment.confirmation_facts(obj)['physical_confirmation_ready']
    experiment.sim.velocity=(.2,0,0)
    assert not experiment.confirmation_facts(obj)['physical_confirmation_ready']


def test_approach_uses_latest_surface_with_confirmation_margin():
    from experiments.legacy.sim import norm, subtract
    experiment=FocusedSearch(config(room='living_room'),None)
    experiment.memory['object_test']={'id':'object_test','room':'living_room','appearance':'An observed target.',
        'visible_surface_xyz':[4,5.5,1.4],'approach_xyz':[1,1,1]}
    point=experiment.options()['approach_object_test']['xyz']
    assert abs(norm(subtract(point,[4,5.5,1.4]))-1.25)<1e-6
    assert norm(subtract(point,[4,5.5,1.4]))+.65<2
    experiment.memory['object_test']['visible_surface_xyz']=[4,6,1.4]
    assert experiment.options()['approach_object_test']['xyz']!=point


def test_task_planner_and_stopping_decision_share_confirmation_contract():
    experiment=FocusedSearch(config(room='living_room'),None)
    obj={'id':'object_test','room':'living_room','appearance':'A clearly observed target.',
         'visible_surface_xyz':[4.1,5.5,1.4],'approach_xyz':[1,1,1],'first_seen_at':0,'last_seen_at':0}
    experiment.memory[obj['id']]=obj;experiment.visible=[obj]
    options=experiment.options()
    planning=experiment.state(options);stopping=experiment.stopping_state(options)
    assert planning['confirmation_distance_m']==stopping['confirmation_distance_m']==2.0
    assert not planning['observed_objects'][0]['physical_confirmation_ready']
    assert not stopping['objects'][0]['physical_confirmation_ready']
    assert '2.15' not in experiment.planning_instructions


def test_deadline_interrupt_requests_jev_status_before_budget_expires():
    gateway=ScriptedGateway(['report_search_incomplete'])
    experiment=FocusedSearch(config(),gateway)
    experiment.sim.time=143
    experiment.task={'choice':'go_search_living_room_high','kind':'go','xyz':[3,6,2.8],
                     'look':'left','started_at':140,'objects_before':set()}
    assert experiment.task_outcome()=='time_budget_review'
    experiment.finish_task('time_budget_review')
    experiment.plan()
    assert experiment.deadline_reviewed
    assert experiment.sim.status=='search_incomplete'
    assert gateway.requests[0][0]['final_budget_review']


def test_control_response_deadline_matches_actuator_lease():
    class SlowPilot:
        def evaluate(self,*unused):
            return {'answers':{'movement':{'choice':'forward'}},'latency_seconds':.6}
    experiment=FocusedSearch(config(),SlowPilot())
    experiment.task={'choice':'go_search_living_room_high','kind':'go','xyz':[3,6,2.8],'description':'Travel to a viewpoint.'}
    experiment.best=10;experiment.progress_at=0
    start=experiment.sim.position
    experiment.move()
    assert experiment.sim.command_lease==.5
    assert not experiment.calls[-1]['accepted']
    assert experiment.sim.action=='brake' and experiment.sim.position==start


def test_high_viewpoint_requires_altitude_in_both_task_and_pilot():
    class Gateway:
        def evaluate(self,state,questions):
            import json
            self.state=json.loads(state.splitlines()[1])
            return {'answers':{'movement':{'choice':'slow_up'}}}
    gateway=Gateway()
    experiment=FocusedSearch(config(),gateway)
    option=experiment.options()['go_search_living_room_north_east_high']
    experiment.task={**option,'started_at':0,'objects_before':set()}
    experiment.sim.position=(*option['xyz'][:2],2.5)
    experiment.sim.time=1;experiment.progress_at=0
    assert not experiment.at_task_position()
    assert experiment.controller_context()['phase']=='travel'
    experiment.pilot.decide(experiment.sim,option['xyz'],experiment.controller_context())
    assert not gateway.state['goal']['inside_arrival_radius']
    assert not gateway.state['flight_status']['position_reached']
    experiment.sim.position=(*option['xyz'][:2],2.74)
    experiment.pilot.decide(experiment.sim,option['xyz'],experiment.controller_context())
    assert gateway.state['goal']['direction']=='up'
    experiment.sim.position=(*option['xyz'][:2],2.8)
    experiment.pilot.look=option['look']
    assert experiment.task_outcome()=='viewpoint_inspected'


def test_high_approach_depends_on_observed_surface_not_hidden_furniture():
    a=FocusedSearch(config(),None)
    b=FocusedSearch(config(variation='mirror_xy'),None)
    a.sim.position=b.sim.position=(9,5,1.4)
    obj={'id':'seen','room':'bedroom','appearance':'An observed object.',
         'visible_surface_xyz':[9,7,2],'approach_xyz':[0,0,0]}
    a.memory['seen']=obj.copy();b.memory['seen']=obj.copy()
    high=a.options()['approach_high_seen']
    assert high==b.options()['approach_high_seen']
    assert high['xyz'][2]==3 and high['altitude_tolerance']==.25


def test_slow_choice_controls_actual_actuator_speed_without_veto():
    class Gateway:
        def evaluate(self,*unused):
            return {'answers':{'movement':{'choice':'slow_forward'}},'latency_seconds':.1}
    experiment=FocusedSearch(config(),Gateway())
    experiment.task={'kind':'go','xyz':[4,5.5,1.4],'description':'Travel.','choice':'test'}
    experiment.best=10;experiment.progress_at=0
    experiment.move()
    assert experiment.sim.command_speed==.3
    assert experiment.calls[-1]['selected_action']=='slow_forward'
    assert experiment.calls[-1]['accepted']
    assert 'slow_forward' in experiment.pilot.questions()['movement']['criteria']


def test_transient_control_error_brakes_then_uses_a_fresh_observation():
    import json
    class Gateway:
        def __init__(self):self.states=[]
        def evaluate(self,state,*unused):
            self.states.append(json.loads(state.splitlines()[1]))
            if len(self.states)==1:return {'error':'ReadTimeout','latency_seconds':3.}
            return {'answers':{'movement':{'choice':'slow_forward'}},'latency_seconds':.1}
    gateway=Gateway();experiment=FocusedSearch(config(),gateway)
    experiment.task={'kind':'go','xyz':[4,5.5,1.4],'description':'Travel.','choice':'test'}
    experiment.best=10;experiment.progress_at=0
    experiment.sim.velocity=(.8,0,0);experiment.sim.command('forward')
    experiment.move()
    assert experiment.sim.status=='running' and experiment.sim.velocity==(0.,0.,0.)
    assert not experiment.calls[-1]['accepted']
    assert experiment.consecutive_api_errors==1
    experiment.move()
    assert experiment.consecutive_api_errors==0
    assert gateway.states[1]['position_metres_xyz']!=gateway.states[0]['position_metres_xyz']
    assert gateway.states[1]['velocity_metres_per_second_xyz']==[0.,0.,0.]
    assert experiment.sim.time>=3.2-1e-6


def test_repeated_api_errors_terminate_without_retrying_forever():
    class Gateway:
        def choose(self,*unused):return {'error':'ReadTimeout','latency_seconds':3.}
    experiment=FocusedSearch(config(),Gateway())
    for count in (1,2):
        experiment.plan()
        assert experiment.sim.status=='running'
        assert experiment.consecutive_api_errors==count
    experiment.plan()
    assert experiment.sim.status=='api_error' and experiment.error=='ReadTimeout'
    assert abs(experiment.sim.time-9)<1e-6


def test_protocol_errors_do_not_retry_and_api_errors_do_not_hide_collisions():
    experiment=FocusedSearch(config(),None)
    experiment.handle_api_result({'error':'ValueError'})
    assert experiment.sim.status=='api_error'
    experiment.sim.status='collision'
    experiment.handle_api_result({'error':'ReadTimeout'})
    assert experiment.sim.status=='collision'
