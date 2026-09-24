import pytest

from experiments.legacy.sim import Box, Scenario, Simulation, norm, ray_box, scenario, subtract


def test_rays_handle_parallel_axes_and_obstacles_behind():
    box = Box((2, 2, 0), (3, 3, 4))
    assert ray_box((0, 2.5, 1), (1, 0, 0), box) == 2
    assert ray_box((0, 1, 1), (1, 0, 0), box) is None
    assert ray_box((4, 2.5, 1), (1, 0, 0), box) is None


def test_ranges_include_drone_size_and_do_not_see_through_walls():
    world = Scenario(
        "test", 0, (1, 5, 1), (10, 5, 1), (Box((3, 4, 0), (4, 6, 4)), Box((6, 4, 0), (7, 6, 4)))
    )
    sim = Simulation(world)
    assert sim.clearance((1, 0, 0)) == pytest.approx(1.8)
    assert sim.clearance((-1, 0, 0)) == pytest.approx(0.8)


def test_swept_collision_catches_thin_obstacles():
    sim = Simulation(Scenario("thin", 0, (1, 5, 1), (10, 5, 1), (Box((1.6, 4, 0), (1.601, 6, 4)),)))
    sim.command("forward")
    sim.advance(0.8)
    assert sim.status == "collision"
    assert sim.position[0] == pytest.approx(1.4)


def test_hover_brakes_over_time_instead_of_stopping_instantly():
    sim = Simulation(scenario("open", 0))
    sim.command("forward")
    sim.advance(0.5)
    x = sim.position[0]
    sim.command("brake")
    sim.advance(0.5)
    assert sim.position[0] - x == pytest.approx(0.25)
    assert norm(sim.velocity) == pytest.approx(0, abs=1e-9)


def test_network_wait_continues_old_command_and_lease_then_brakes():
    sim = Simulation(scenario("open", 0))
    sim.command("forward")
    sim.advance(2.0)
    assert sim.position[0] == pytest.approx(sim.world.start[0] + 0.8)
    assert norm(sim.velocity) == pytest.approx(0, abs=1e-9)
    assert sim.time == pytest.approx(2.0)


def test_goal_requires_stopping_not_flying_through():
    world = Scenario("test", 0, (1, 5, 1), (1.3, 5, 1), ())
    moving = Simulation(world, velocity=(1, 0, 0))
    moving.command("forward")
    moving.advance(0.5)
    assert moving.status == "running"
    resting = Simulation(world)
    resting.advance(0.5)
    assert resting.status == "success"


def test_vertical_barrier_cannot_be_crossed_at_start_altitude():
    sim = Simulation(scenario("overpass", 0))
    for _ in range(8):
        sim.command("forward")
        sim.advance(0.6)
    assert sim.status == "collision"


def test_observation_does_not_expose_hidden_map_or_route():
    observation = Simulation(scenario("u_trap", 0)).observation([])
    assert "obstacles" not in observation
    assert "route" not in observation
    assert "scenario" not in observation
    assert len(observation["ranges"]) == 10


def test_heldout_seed_is_repeatable_and_changes_geometry():
    assert scenario("wall", 1) == scenario("wall", 1)
    assert scenario("wall", 0).obstacles != scenario("wall", 1).obstacles


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", ["open", "pillar", "wall", "overpass", "u_trap"])
def test_all_courses_are_physically_reachable_with_manual_waypoints(name, seed):
    """Privileged feasibility check only; these waypoints never reach a policy.

    Commands are refreshed at physics rate, so this establishes geometric and
    actuator feasibility, not an equal-information or equal-latency baseline.
    """
    sim = Simulation(scenario(name, seed))
    start, goal = sim.world.start, sim.world.goal
    if name == "open":
        waypoints = [goal]
    elif name == "overpass":
        waypoints = [(start[0], start[1], 3.1), (goal[0], goal[1], 3.1), goal]
    elif name == "u_trap":
        exit_x = 10.0 if seed % 2 else 2.0
        waypoints = [(exit_x, start[1], 1.0), (exit_x, 8.9, 1.0), (goal[0], 8.9, 1.0), goal]
    else:
        waypoints = [(start[0], 8.9, 1.0), (goal[0], 8.9, 1.0), goal]
    for waypoint in waypoints:
        for _ in range(2000):
            offset = subtract(waypoint, sim.position)
            if norm(offset) < 0.35:
                sim.command("brake")
                if norm(sim.velocity) < 0.01:
                    break
            else:
                axis = max(range(3), key=lambda i: abs(offset[i]))
                sim.command(
                    ["forward", "left", "up"][axis]
                    if offset[axis] > 0
                    else ["back", "right", "down"][axis]
                )
            sim.advance(0.02)
            assert sim.status != "collision"
        else:
            pytest.fail("Manual route failed to reach waypoint")
    sim.command("brake")
    sim.advance(0.5)
    assert sim.status == "success"
