import json
from dataclasses import replace
from types import SimpleNamespace

import httpx

from experiments.legacy.search import deepseek_ask, episode, planner_state
from experiments.legacy.search_world import (
    Home,
    Object,
    evaluate_report,
    make_home,
    room_at,
    search_waypoints,
    update_memory,
    visible_objects,
)
from experiments.legacy.sim import Box, Scenario, Simulation


def sensor_scene(obstacles=(), box=None):
    box = box or Box((3, 1, 0), (3.5, 2, 2.5))
    obj = Object("opaque_7", "bookcase", box, "Shelves containing rows of bound volumes.")
    world = Scenario(
        "sensor", 0, (1.0, 1.5, 1.0), (-100.0, -100.0, -100.0), (*obstacles, box), (10.0, 10.0, 4.0)
    )
    return Home(world, (obj,), obj.id)


def test_visible_object_exposes_appearance_not_privileged_label_or_bounds():
    home = sensor_scene()
    seen = visible_objects(home, home.world.start)
    assert len(seen) == 1
    assert seen[0]["id"] == "opaque_7"
    assert set(seen[0]) == {"id", "appearance", "visible_surface_xyz", "range_metres", "room"}
    assert seen[0]["visible_surface_xyz"][0] == 3


def test_solid_wall_occludes_semantic_sensor():
    home = sensor_scene((Box((2, 0, 0), (2.2, 4, 4)),))
    assert visible_objects(home, home.world.start) == []


def test_range_limit_hides_object():
    home = sensor_scene(box=Box((7, 1, 0), (7.5, 2, 2.5)))
    assert visible_objects(home, home.world.start) == []


def test_low_obstacle_can_be_seen_over_in_3d():
    home = sensor_scene((Box((2, 0, 0), (2.2, 4, 0.75)),))
    assert visible_objects(home, (1.0, 1.5, 1.5))


def test_unseen_target_relocation_does_not_change_model_input():
    states = []
    for room in ["bedroom", "study"]:
        home = make_home(9, room)
        sim = Simulation(home.world)
        seen = visible_objects(home, sim.position)
        assert home.target_id not in {o["id"] for o in seen}
        memory = {}
        update_memory(memory, seen, sim.position, 0.0)
        state, criteria, _ = planner_state(
            "Find a bookcase", sim, memory, seen, [], [], None, "starting", "none"
        )
        states.append(json.dumps([state, criteria], sort_keys=True))
    assert states[0] == states[1]


def test_viewpoints_do_not_depend_on_hidden_furniture():
    points = search_waypoints()
    for point in points.values():
        assert room_at(point["xyz"]) == point["room"]
    assert len(points) == 19


def test_memory_remembers_only_seen_objects_and_preserves_initial_approach():
    home = sensor_scene()
    memory = {}
    update_memory(memory, visible_objects(home, home.world.start), home.world.start, 1.0)
    approach = memory[home.target_id]["approach_xyz"]
    update_memory(memory, [], (1.0, 9.0, 1.0), 2.0)
    assert memory[home.target_id]["last_seen_at"] == 1.0
    position = (1.2, 1.5, 1.0)
    update_memory(memory, visible_objects(home, position), position, 3.0)
    assert memory[home.target_id]["first_seen_at"] == 1.0
    assert memory[home.target_id]["approach_xyz"] == approach


def test_report_requires_correct_identity_visibility_proximity_and_stopping():
    home = sensor_scene()
    sim = Simulation(home.world)
    seen = visible_objects(home, sim.position)
    assert evaluate_report("report_found_opaque_7", home, sim, seen) == "success"
    assert evaluate_report("report_found_other", home, sim, seen) == "wrong_object"
    assert evaluate_report("report_found_opaque_7", home, sim, []) == "unverified_report"
    sim.velocity = (1, 0, 0)
    assert evaluate_report("report_found_opaque_7", home, sim, seen) == "premature_report"
    sim.velocity = (0, 0, 0)
    sim.position = (0.3, 1.5, 1)
    assert evaluate_report("report_found_opaque_7", home, sim, seen) == "premature_report"
    assert evaluate_report("report_not_found", home, sim, seen) == "false_absence"
    assert (
        evaluate_report("report_not_found", replace(home, target_id=None), sim, seen)
        == "correct_absence"
    )


def test_recognition_that_exhausts_time_does_not_start_another_api_call(monkeypatch):
    calls = []

    def fake_ask(client, state, instructions, criteria):
        calls.append(state)
        return {"choice": "none", "latency_seconds": 2.0, "state": state}

    monkeypatch.setattr("experiments.legacy.search.ask", fake_ask)
    args = SimpleNamespace(
        seed=0,
        target_room="bedroom",
        absent=False,
        seconds=1.0,
        budget=0.08,
        planner="jev",
        mission="Find a bookcase",
    )
    result = episode(args, SimpleNamespace(client=None))
    assert result["status"] == "timeout"
    assert len(calls) == 1
    assert result["calls"] == 1
    assert result["simulation_seconds"] == 1.0


def test_empty_chat_completion_retains_usage_and_finish_reason():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "model": "test",
                "usage": {"cost": 0.001},
                "choices": [{"finish_reason": "length", "message": {"content": None}}],
            },
        )
    )
    with httpx.Client(transport=transport) as client:
        result = deepseek_ask(client, {}, "choose", {"a": "Option a"})
    assert result["error"] == "missing_structured_content"
    assert result["finish_reason"] == "length"
    assert result["usage"]["cost"] == 0.001
