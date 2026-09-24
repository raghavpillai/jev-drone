import numpy as np
import pytest

pytest.importorskip("pybullet")

from jev_drone.perception.motion import MotionTracker
from jev_drone.perception.sensors import Frame, Sensors, basis, detect_markers, swept_clearance
from jev_drone.planning.agent import Experiment
from jev_drone.world.world import Config, World


def synthetic_frame(depth=4.0):
    forward, right, up = basis((1, 0, 0))
    return Frame(
        0.0,
        np.zeros((72, 96, 3), np.uint8),
        np.full((72, 96), depth),
        np.zeros(3),
        forward,
        right,
        up,
        43.0,
    )


def test_camera_depth_clearance_accounts_for_body_width_and_unknown():
    frame = synthetic_frame()
    assert swept_clearance(frame, np.zeros(3), (1, 0, 0)) >= 2.5
    assert swept_clearance(frame, np.zeros(3), (-1, 0, 0)) is None
    # Obstacle at a footprint edge blocks even with clear central pixel.
    frame.depth[32:40, 65:76] = 0.7
    assert swept_clearance(frame, np.zeros(3), (1, 0, 0)) < 0.4
    frame.depth[:] = np.nan
    assert swept_clearance(frame, np.zeros(3), (1, 0, 0)) is None


def test_marker_detection_uses_pixels_and_measured_depth():
    frame = synthetic_frame(2.0)
    frame.rgb[32:40, 44:52] = (210, 12, 12)
    detections = detect_markers(frame)
    assert len(detections) == 1
    assert detections[0]["marker"] == "red"
    assert abs(detections[0]["position"][0] - 2.0) < 0.01
    frame.rgb[:] = 0
    assert detect_markers(frame) == []
    frame.rgb[32:40, 44:52] = (120, 75, 25)  # Brown furniture is not a yellow marker.
    assert detect_markers(frame) == []


def test_renderer_occludes_target_and_exposes_no_segmentation():
    world = World(Config(depth_noise=0, dropout=0, wind=0, odometry_drift=0))
    try:
        sensors = Sensors(world)
        sensors.advance(0.15)
        observation = sensors.observation()
        assert observation["frame_age_seconds"] >= 0.1 - 1e-6
        assert "red" not in [d["marker"] for d in observation["detections"]]
        assert "body" not in str(observation)
        assert observation["clearance_m"]["west"] is None
        assert observation["clearance_m"]["east"] is not None
    finally:
        world.close()


def test_camera_turns_gradually_and_expired_command_brakes():
    world = World(Config(wind=0))
    try:
        world.command("look_north")
        world.advance(0.25)
        assert 0.4 < world.camera_direction[1] < 0.6
        world.advance(0.6)
        assert world.camera_direction[1] > 0.999
        world.command("cruise_east")
        world.advance(0.5)
        assert world.velocity[0] > 0.35
        world.advance(1.5)
        assert np.linalg.norm(world.velocity) < 0.02
        assert world.status == "running"
    finally:
        world.close()


def test_blackout_remains_unknown_and_noise_is_reproducible():
    frames = []
    for _ in range(2):
        world = World(Config(seed=2501, case="blackout"))
        try:
            sensors = Sensors(world)
            frames.append(sensors.queue[0][0].depth.copy())
            world.time = 13
            sensors.capture()
            sensors.advance(0.2)
            observation = sensors.observation()
            assert all(value is None for value in observation["clearance_m"].values())
            assert observation["detections"] == []
        finally:
            world.close()
    np.testing.assert_equal(frames[0], frames[1])


def test_jev_detour_sets_intent_without_automatic_flight():
    class Gateway:
        def choose(self, state, instructions, criteria):
            choice = "detour_north" if state["active_detour"] is None else "brake"
            assert choice in criteria
            return {"choice": choice, "latency_seconds": 0.1, "state": state}

    experiment = Experiment(Config(policy="v3", wind=0), Gateway())
    try:
        experiment.task = {"name": "test", "position": [5, 2, 1.3]}
        experiment.control(experiment.observe())
        assert experiment.detour["selected_by"] == "jev"
        assert np.linalg.norm(experiment.world.velocity) < 0.001
        experiment.control(experiment.observe())
        assert np.linalg.norm(experiment.world.velocity) < 0.001
    finally:
        experiment.world.close()


def test_observed_motion_predicts_crossing_and_static_tracks_do_not():
    tracker = MotionTracker()
    for t in np.arange(0, 1.0, 0.125):
        tracker.update(
            t,
            [
                {
                    "color_bin": 2,
                    "position": np.array([1.2, 0.65 - 0.5 * t, 1.0]),
                    "half_extent": np.array([0.2, 0.2, 0.7]),
                    "pixels": 30,
                }
            ],
        )
    observation = tracker.observation(np.array([0.0, 0.0, 1.0]), np.array([0.6, 0, 0]), 0.6, 0.875)
    assert observation["current_course_risk"]
    assert observation["predicted_risk_by_direction"]["east"]
    assert not observation["predicted_risk_by_direction"]["west"]
    assert abs(observation["moving_visual_tracks"][0]["velocity_estimate"][1] + 0.5) < 0.02
    tracker = MotionTracker()
    for t in np.arange(0, 1.0, 0.125):
        tracker.update(
            t,
            [
                {
                    "color_bin": 2,
                    "position": np.array([1.2, 0.2, 1.0]),
                    "half_extent": np.array([0.2, 0.2, 0.7]),
                    "pixels": 30,
                }
            ],
        )
    assert not tracker.observation(np.zeros(3), np.zeros(3), 0.6, 0.875)["moving_visual_tracks"]


def test_integral_stabilizer_reduces_wind_drift_without_navigation():
    drift = {}
    for servo in ("p", "pi"):
        world = World(Config(wind=0.6, servo=servo))
        try:
            world.command("brake")
            world.advance(12.0)
            drift[servo] = np.linalg.norm(world.position - world.start)
            assert world.status == "running"
        finally:
            world.close()
    assert drift["pi"] < drift["p"] * 0.5
