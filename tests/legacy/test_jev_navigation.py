import json
from types import SimpleNamespace

import httpx
import pytest

from experiments.legacy.all_jev_experiment import AllJevExperiment
from jev_drone.gateway import JevGateway, validate_answers


def test_gateway_persists_completed_calls_without_credentials(tmp_path):
    import json
    import httpx
    journal=tmp_path/'calls.jsonl'
    gateway=JevGateway('openrouter','secret-not-for-log',journal=journal,
        transport=httpx.MockTransport(lambda request:httpx.Response(503)))
    try:
        result=gateway.evaluate({'observation':'test'},{})
    finally:
        gateway.close()
    assert json.loads(journal.read_text())==result
    assert result['error']=='http_503'
    assert 'secret-not-for-log' not in journal.read_text()


def test_recorded_inputs_do_not_gain_future_search_memories(tmp_path):
    journal = tmp_path/'calls.jsonl'
    memory = {"seen": ["red"]}
    questions = {"selection": {"criteria": {"wait": "Wait."}}}
    gateway = JevGateway('openrouter', 'test', journal=journal,
                         transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    try:
        result = gateway.evaluate({"memory": memory}, questions)
    finally:
        gateway.close()
    memory['seen'].append('blue')
    questions['selection']['criteria']['move'] = 'Move.'
    assert result['state']['memory']['seen'] == ['red']
    assert result == json.loads(journal.read_text())
from experiments.legacy.jev_navigation import Navigation
from experiments.legacy.observed_map import ObservedMap
from experiments.legacy.representation_experiment import world
from experiments.legacy.search_world import make_home, visible_objects
from experiments.legacy.sim import Simulation


def test_map_does_not_fill_missing_or_beyond_range():
    memory=ObservedMap()
    memory.record((1,1,1),{"forward":2.},0)
    assert memory.evidence((1.5,1,1))=="free_evidence"
    assert memory.evidence((3,1,1))=="occupied_evidence"
    assert memory.evidence((4,1,1))=="unknown"
    assert memory.evidence((1,2,1))=="unknown"


def test_map_retains_conflicting_measurements():
    memory=ObservedMap()
    memory.record((1,1,1),{"forward":1.},0)
    memory.record((1,1,1),{"forward":3.},1)
    assert memory.evidence((2,1,1))=="conflicting"


def test_limited_sensor_and_look_are_model_decisions():
    sim=Simulation(world("doorway",0))
    nav=Navigation(None,limited=True)
    common,ranges=nav.observation(sim,sim.world.goal)
    assert ranges["back"] is None and ranges["up"] is None
    assert ranges["forward"] is not None
    nav.accept(sim,{"action":"brake","intent":"direct","look":"back"},True,sim.world.goal)
    _,ranges=nav.observation(sim,sim.world.goal)
    assert ranges["back"] is not None and ranges["forward"] is None


def test_delayed_frame_uses_historical_camera_direction():
    sim=Simulation(world("doorway",0))
    nav=Navigation(None,limited=True,degraded=True)
    sim.advance(.5)
    nav.accept(sim,{"action":"brake","intent":"direct","look":"back"},True,sim.world.goal)
    common,ranges=nav.observation(sim,sim.world.goal)
    assert common["camera"]["captured_look_direction"]=="forward"
    assert ranges["back"] is None


@pytest.mark.parametrize("provider",["openrouter","vercel"])
def test_gateway_uses_provider_specific_endpoint_and_validates_choice(provider):
    def handler(request):
        body=json.loads(request.content)
        assert "jev" in body["model"]
        assert request.url.path==("/v1/evaluate" if provider=="vercel" else "/api/alpha/decisions")
        return httpx.Response(200,json={"model":body["model"],"answers":{"selection":{"choice":"yes","probabilities":{"yes":1,"no":0}}},
            "usage":{},"providerMetadata":{"gateway":{"cost":"0"}}})
    gateway=JevGateway(provider,"test",transport=httpx.MockTransport(handler))
    try:
        result=gateway.choose("state","question",{"yes":"yes","no":"no"})
        assert result["choice"]=="yes" and not result.get("error")
        if provider=="vercel": assert result["usage"]["cost"]==0
    finally:
        gateway.close()


def test_invalid_answer_fails_closed():
    with pytest.raises(ValueError):
        validate_answers({"q":{"choice":"forward","probabilities":{"forward":float('nan')}}},
                         {"q":{"criteria":{"forward":"go"}}})


def test_camera_filter_never_reveals_objects_outside_cone():
    home=make_home(1)
    position=home.world.start
    observations=visible_objects(home,position,view_direction=(0,-1,0))
    assert observations
    assert all(o["visible_surface_xyz"][1]<position[1] for o in observations)


def test_all_roles_use_same_jev_gateway():
    calls=[]
    class Gateway:
        def choose(self,*args):
            calls.append(args)
            return {"choice":"yes","latency_seconds":.01,"model":"typesafe/jev-test"}
    args=SimpleNamespace(seed=1,case="near",architecture="task",planner="jev",seconds=30,budget=.2,
                         memory="recovery",limited=True,uncertain=True,degraded=False)
    experiment=AllJevExperiment(args,Gateway())
    for role in ["planner","recognition","navigator","recovery"]:
        assert experiment.request(role,{},"",{"yes":"yes"})=="yes"
    assert len(calls)==4


def test_inspection_handoff_contains_only_remembered_target():
    from experiments.legacy.inspection_experiment import InspectionNavigation
    class Gateway:
        def evaluate(self,state,questions):
            self.state=state
            return {"error":"test_no_network"}
    gateway=Gateway()
    sim=Simulation(world("doorway",0))
    nav=InspectionNavigation(gateway,limited=True)
    nav.decide(sim,sim.world.goal)
    assert "target_observation" not in gateway.state
    nav.observed_target={"appearance":"observed shelving","last_seen_at":2.,"visible_surface_xyz":[2.,7.,1.4]}
    nav.decide(sim,sim.world.goal)
    assert "target_observation" in gateway.state and "surface_offset_forward_left_up_m" in gateway.state
    assert "obstacles" not in gateway.state and "ground_truth" not in gateway.state


def test_joint_scan_preserves_raw_jev_choice_and_only_executes_selected_action():
    from experiments.legacy.scanning_experiment import ScanningNavigation
    nav=ScanningNavigation(None,limited=True)
    assert "look" not in nav.questions()
    assert len(nav.questions()["movement"]["criteria"])==13
    result=nav.decode({"answers":{"movement":{"choice":"look_up"},"intent":{"choice":"direct"}}})
    assert result["action"]=="brake" and result["look"]=="up"
    assert result["answers"]["movement"]["choice"]=="look_up"
    result=nav.decode({"answers":{"movement":{"choice":"forward"},"intent":{"choice":"direct"}}})
    assert result["action"]=="forward" and result["look"]=="forward"
