from dataclasses import replace
from types import SimpleNamespace

from experiments.legacy.active_search import ActiveSearch, mission_scene
from experiments.legacy.search_attention import PROBES, SearchAttention, visible_probes
from experiments.legacy.sim import Box, Scenario


def args(memory="coverage"):
    return SimpleNamespace(
        seed=501,
        case="near",
        item="bookcase",
        room="living_room",
        memory=memory,
        seconds=120,
        budget=0.15,
    )


def test_occlusion_does_not_mark_hidden_region_as_seen():
    world = Scenario(
        "test",
        0,
        (1.0, 5.0, 1.4),
        (-100.0,) * 3,
        (Box((2.0, 0.0, 0.0), (2.2, 10.0, 4.0)),),
        (18.0, 10.0, 4.0),
    )
    visible = visible_probes(world, world.start, "forward")
    assert all(PROBES[p]["xyz"][0] < 2 for p in visible)
    unobstructed = visible_probes(replace(world, obstacles=()), world.start, "forward")
    assert set(unobstructed) - set(visible)


def test_all_looks_at_one_pose_do_not_exhaust_room_and_memory_survives_reentry():
    world = Scenario("test", 0, (1.0, 5.0, 1.4), (-100.0,) * 3, (), (18.0, 10.0, 4.0))
    attention = SearchAttention()
    for i, look in enumerate(("forward", "back", "left", "right", "up", "down")):
        attention.record(
            visible_probes(world, world.start, look), world.start, look, i, "living_room"
        )
    first = attention.summary("living_room")
    assert 0 < first["fraction"] < 1
    attention.record([], (7.0, 5.0, 1.4), "forward", 7, "bedroom")
    event = attention.record(
        visible_probes(world, world.start, "forward"), world.start, "forward", 8, "living_room"
    )
    assert attention.summary("living_room") == first
    assert event["repeat_view"] and event["new_visible_samples"] == 0


def test_generic_mission_does_not_leak_kind_or_target_id():
    experiment = ActiveSearch(args(), None)
    experiment.observe()
    state = experiment.state(experiment.options())
    assert "target_id" not in state and "ground_truth" not in state
    assert all("kind" not in o and "box" not in o for o in state["observed_objects"])
    assert {o["id"] for o in state["observed_objects"]} == set(experiment.memory)
    for item in ("plant", "cabinet", "table", "bookcase"):
        scene = mission_scene(501, "near", item, "living_room")
        assert next(o.kind for o in scene.home.objects if o.id == scene.home.target_id) == item
    assert mission_scene(501, "near", "bed", "living_room").home.target_id is None


def test_arrival_without_camera_verification_does_not_complete_approach():
    experiment = ActiveSearch(args(), None)
    experiment.task = {
        "kind": "approach",
        "xyz": experiment.sim.position,
        "object": "object_x",
        "started_at": 0,
        "objects_before": set(),
    }
    experiment.progress_at = 0
    experiment.sim.time = 1
    assert experiment.task_outcome() is None
    experiment.visible = [{"id": "object_x"}]
    assert experiment.task_outcome() == "arrived_and_object_visible"


def test_failed_attempts_and_coverage_are_explicit_and_ablatable():
    experiment = ActiveSearch(args(), None)
    experiment.observe()
    experiment.task = {
        "choice": "look_forward",
        "started_at": 0,
        "seen_before": len(experiment.attention.seen),
    }
    experiment.finish_task("stalled_choose_another_view_or_route")
    state = experiment.state(experiment.options())
    assert state["task_counts"]["look_forward"] == 1
    assert state["recent_task_results"][0]["new_visible_samples"] == 0
    assert "search_attention" in state
    experiment.args.memory = "checklist"
    assert "search_attention" not in experiment.state(experiment.options())


def test_potential_never_consults_hidden_obstacles():
    # Planner upper bound uses known room coordinates and actually recorded sightings only.
    attention = SearchAttention()
    before = attention.potential((2, 5, 1.4), "forward", "living_room")
    sample = next(
        p
        for p in PROBES
        if p
        in visible_probes(
            Scenario("test", 0, (2, 5, 1.4), (-100.0,) * 3, (), (18, 10, 4)), (2, 5, 1.4), "forward"
        )
    )
    attention.record([sample], (2, 5, 1.4), "forward", 0, "living_room")
    assert attention.potential((2, 5, 1.4), "forward", "living_room") == before - 1


def test_recorded_planner_input_does_not_gain_future_room_entries():
    experiment = ActiveSearch(args(), None)
    experiment.observe()
    state = experiment.state(experiment.options())
    experiment.entries.append({"room": "bedroom", "time": 10})
    assert len(state["room_entries"]) == 1


def test_correct_object_report_requires_clear_perceived_evidence():
    experiment = ActiveSearch(args(), None)
    obj = next(o for o in experiment.scene.home.objects if o.id == experiment.scene.home.target_id)
    observation = {
        "id": obj.id,
        "room": "living_room",
        "appearance": "A tall wooden storage unit; shelving and contents are unclear from this view.",
        "visible_surface_xyz": list(obj.box.lo),
        "approach_xyz": [obj.box.lo[0], 7.5, 1.4],
        "first_seen_at": 0,
        "last_seen_at": 0,
    }
    experiment.memory[obj.id] = observation
    experiment.visible = [observation]

    class Reporter:
        def choose(self, *unused):
            return {"choice": "report_found_" + obj.id, "latency_seconds": 0.1}

    experiment.gateway = Reporter()
    experiment.plan()
    assert experiment.sim.status == "unverified_report"


def test_anywhere_mission_accepts_matching_object_in_each_room():
    from experiments.legacy.search_world import update_memory, visible_objects

    for room, x in (("living_room", 5.1), ("bedroom", 11.1), ("study", 17.1)):
        config = args()
        config.item = "plant"
        config.room = "anywhere"
        experiment = ActiveSearch(config, None)
        experiment.sim.position = (x, 6.5, 1.4)
        experiment.pilot.look = "left"
        experiment.visible = visible_objects(
            experiment.scene.home, experiment.sim.position, view_direction=(0, 1, 0)
        )
        update_memory(experiment.memory, experiment.visible, experiment.sim.position, 0)
        target = next(
            o
            for o in experiment.scene.home.objects
            if o.kind == "plant" and o.id in experiment.memory
        )

        class Reporter:
            def choose(self, *unused):
                return {"choice": "report_found_" + target.id, "latency_seconds": 0.1}

        experiment.gateway = Reporter()
        experiment.plan()
        assert experiment.sim.status == "success", room
