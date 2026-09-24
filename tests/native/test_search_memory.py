from jev_drone.planning.search_memory import SearchMemory


def test_viewpoint_memory_requires_front_observation_and_keeps_altitudes_separate():
    memory = SearchMemory()
    observation = dict(room="bedroom", position_estimate=[7.5, 1., 1.3], yaw_degrees=-45.,
                       front_frame_available=False, valid_depth_fraction=.9)
    memory.observe(observation, 1.)
    assert not any(v["observed_near_viewpoint"] for v in memory.summary("bedroom").values())
    observation["front_frame_available"] = True
    memory.observe(observation, 2.)
    views = memory.summary("bedroom")
    assert views["view_bedroom_0_0_1.3"]["observed_heading_degrees"] == [315]
    assert not views["view_bedroom_0_0_2.9"]["observed_near_viewpoint"]
    assert not views["view_bedroom_1_0_1.3"]["observed_near_viewpoint"]


def test_memory_describes_search_without_selecting_or_removing_tasks():
    memory = SearchMemory()
    state = {"sensors": {"room": "bedroom"}, "options": {
        "view_bedroom_0_0_1.3": {"position": [7.5, 1., 1.3]}, "give_up": {"purpose": "terminate"}}}
    criteria = {name: "Original." for name in state["options"]}
    described, choices = memory.describe(state, criteria)
    assert set(choices) == set(criteria)
    assert described["options"]["view_bedroom_0_0_1.3"]["position"] == [7.5, 1., 1.3]
    assert not described["options"]["view_bedroom_0_0_1.3"]["observed_near_viewpoint"]
    assert "observed_near_viewpoint" not in state["options"]["view_bedroom_0_0_1.3"]
