import json
from types import SimpleNamespace

import pytest

from experiments.legacy.architecture_experiment import Experiment
from experiments.legacy.experiment_world import make_scene
from experiments.legacy.search_world import visible_objects
from experiments.legacy.sim import Simulation, norm
from experiments.legacy.task_planning import all_checked, planning_state, room_coverage, task_options, waypoint_options


def args(**overrides):
    values = dict(seed=101,case="far",architecture="task",planner="deepseek",seconds=210.,budget=.25)
    return SimpleNamespace(**(values|overrides))


def test_non_planner_roles_cannot_call_deepseek(monkeypatch):
    called=[]
    def fake(name):
        def request(*a,**kw):
            called.append(name)
            return {"choice":"a","latency_seconds":.01}
        return request
    monkeypatch.setattr("experiments.legacy.architecture_experiment.ask",fake("jev"))
    monkeypatch.setattr("experiments.legacy.architecture_experiment.deepseek_ask",fake("deepseek"))
    experiment=Experiment(args(),SimpleNamespace(client=None))
    for role in ["recognition","navigator","planner"]:
        experiment.request(role,{},"",{"a":"a"})
    assert called==["jev","jev","deepseek"]


def test_movement_uses_local_jev_decision_not_planner(monkeypatch):
    def forbidden(*a,**kw):
        raise AssertionError("Planner called for movement")
    monkeypatch.setattr("experiments.legacy.architecture_experiment.deepseek_ask",forbidden)
    class Local:
        def decide(self, observation):
            return {"action":"up","intent":"direct","latency_seconds":.3,"model":"typesafe/jev-test"}
    experiment=Experiment(args(),Local())
    experiment.target=(2.,5.5,2.8)
    experiment.target_name="test"
    experiment.move()
    assert experiment.sim.action=="up"
    assert experiment.calls[-1]["role"]=="control"
    assert experiment.calls[-1]["model"].startswith("typesafe/jev")


def test_task_contract_contains_objective_completion_and_feedback():
    tasks,criteria=task_options({})
    task=tasks["search_bedroom"]
    assert task["kind"]=="search_room"
    assert task["room"]=="bedroom"
    assert task["complete_when"] and task["report_on"]
    assert not any(k.startswith("forward") for k in criteria)


def test_task_scopes_navigation_without_hidden_obstacle_filtering():
    scene=make_scene(101,"far")
    sim=Simulation(scene.home.world)
    tasks,_=task_options({})
    points,_,criteria=waypoint_options(scene,sim,{},set(),[],tasks["search_bedroom"])
    assert len([k for k in points if k.startswith("search_")])==5
    assert "search_study_high" not in points
    assert "door_1_study" in points
    assert "report_blocked" in criteria


def test_absence_requires_checked_views_not_attempts():
    scene=make_scene(104,"absent")
    assert not all_checked(scene,set())
    assert all_checked(scene,set(scene.points))
    experiment=Experiment(args(seed=104,case="absent"),SimpleNamespace(client=None))
    experiment.finish("report_not_found")
    assert experiment.sim.status=="unsupported_absence"


def test_new_initial_planner_state_hides_unseen_object_geometry():
    scene=make_scene(101,"far")
    sim=Simulation(scene.home.world)
    state=planning_state(scene,sim,{},visible_objects(scene.home,sim.position),set(),"none",{})
    encoded=json.dumps(state)
    assert scene.home.target_id not in {o["id"] for o in state["visible_now"]}
    assert "ground_truth" not in state and "obstacles" not in state
    assert "bound volumes" not in encoded


def test_speed_choice_is_executed_without_changing_max_speed():
    sim=Simulation(make_scene(1,"near").home.world)
    sim.command("up",speed=.4)
    sim.advance(.5)
    assert norm(sim.velocity)==pytest.approx(.4)
    assert sim.speed==1.
    sim.command("up")
    sim.advance(.5)
    assert norm(sim.velocity)==pytest.approx(1.)
    for speed in [-1,2,float("nan")]:
        with pytest.raises(ValueError):
            sim.command("forward",speed=speed)
