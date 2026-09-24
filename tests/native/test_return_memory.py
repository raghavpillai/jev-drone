from jev_drone.planning.return_memory import ReturnMemory


def observation(x, y=2., z=1.3):
    return {"position_estimate": [x, y, z], "room": "living", "yaw_degrees": 0.}


def test_return_uses_observed_outbound_positions_and_does_not_record_retrace_loops():
    memory = ReturnMemory()
    for x in (1., 2., 3., 4.):
        memory.observe(observation(x), False)
    memory.observe(observation(4.1), True)
    memory.observe(observation(3.), True)
    options = memory.options(observation(3.), [])
    assert len(memory.points) == 4
    assert options["retrace_1"]["position"] == [2., 2., 1.3]
    assert "retrace_2" not in options  # Already at this position.
    completed = [{"return_route_index": 1, "outcome": "arrived"}]
    assert set(memory.options(observation(2.), completed)) == {"retrace_0"}


def test_failed_waypoint_does_not_advance_return_and_altitude_is_preserved():
    memory = ReturnMemory()
    memory.observe(observation(1., z=2.8), False)
    memory.observe(observation(2., z=2.8), False)
    memory.observe(observation(3., z=2.8), True)
    options = memory.options(observation(3., z=2.8), [{"return_route_index": 1, "outcome": "stalled"}])
    assert options["retrace_1"]["position"][2] == 2.8
    assert len(options) == 2


def test_history_is_bounded_and_no_options_are_exposed_during_outbound_search():
    memory = ReturnMemory()
    for x in range(600):
        memory.observe(observation(float(x)), False)
    assert len(memory.points) <= 256
    assert memory.options(observation(598.), []) == {}
