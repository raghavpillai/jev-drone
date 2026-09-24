import json

from experiments.legacy.representation_experiment import DIRECTIONS, encode, observe, world
from experiments.legacy.sim import Simulation


def test_encodings_preserve_measurements_and_unknowns():
    sim = Simulation(world("overpass", 0))
    common, readings = observe(sim, [], "direct", True, 0)
    assert len(readings) == len(DIRECTIONS) == 26
    assert None in readings.values()
    grid = encode(common, readings, "grid")["directional_depth_grid"]
    cells = [v for layer in grid.values() for row in layer for v in row if v != "SELF"]
    assert sorted(cells, key=str) == sorted(readings.values(), key=str)
    assert len(encode(common, readings, "facts")["spatial_facts"]) == 26
    assert encode(common, readings, "numeric")["directional_clearance_metres"] == readings
    brief = encode(common, readings, "brief")
    assert brief.index("forward:") < brief.index("SUPPLEMENTARY")
    assert brief.count("clearance ") == sum(v is not None for v in readings.values())
    assert brief.count("UNKNOWN") == sum(v is None for v in readings.values())


def test_delayed_observations_use_past_pose_and_preserve_unknown():
    sim = Simulation(world("doorway", 0))
    sim.command("forward")
    sim.advance(0.7)
    common, readings = observe(sim, [], "direct", True, 0)
    assert common["sensor"]["age_seconds"] >= 0.3 - 1e-3
    assert common["sensor"]["capture_position"][0] < sim.position[0]
    assert all(v is None or 0 <= v <= 3 for v in readings.values())
    for variant in ("numeric", "facts", "grid", "brief"):
        text = json.dumps(encode(common, readings, variant))
        assert "obstacles" not in text and "world" not in text


def test_unobserved_geometry_cannot_affect_readings():
    from dataclasses import replace

    from experiments.legacy.sim import Box

    base = world("doorway", 0)
    hidden = Box((10, 1, 0), (11, 2, 3))
    a = observe(Simulation(base), [], "direct", False, 0)
    b = observe(
        Simulation(replace(base, obstacles=base.obstacles + (hidden,))), [], "direct", False, 0
    )
    assert a == b
