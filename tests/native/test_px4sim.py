"""Contracts whose mistakes can silently invalidate flight experiments."""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pymavlink", reason="Run PX4 integration-unit tests in the simulator image")
from jev_drone.sim.flight import attitude_enu, enu_to_ned
from jev_drone.sim.scene import generate
from jev_drone.world.world import Config


def test_enu_ned_axis_signs_and_body_orientation():
    assert np.allclose(enu_to_ned([2, 3, 4]), [3, 2, -4])
    assert np.allclose(enu_to_ned(enu_to_ned([2, 3, 4])), [2, 3, 4])
    # PX4 heading 90 degrees (east) is identity for an ENU/FLU camera.
    assert np.allclose(attitude_enu(0, 0, np.pi / 2), np.eye(3), atol=1e-10)
    assert np.allclose(attitude_enu(0, 0, 0)[:, 0], [0, 1, 0])
    assert np.allclose(np.linalg.det(attitude_enu(0.2, -0.1, 0.4)), 1)


def test_real_simulator_is_not_advanced_twice_for_network_latency():
    from jev_drone.planning.agent import Experiment

    class Gateway:
        def choose(self, *args):
            return {"latency_seconds": 0.4, "usage": {}, "choice": "brake"}

    for realtime in (False, True):
        experiment = object.__new__(Experiment)
        experiment.world = SimpleNamespace(time=1.0, realtime=realtime)
        advances = []
        experiment.sensors = SimpleNamespace(advance=advances.append)
        experiment.gateway = Gateway()
        experiment.cost, experiment.calls, experiment.error_streak = 0, [], 0
        experiment.request("control", {}, "", {"brake": "stop"})
        assert advances == ([] if realtime else [0.4])


def test_scene_keeps_native_motors_contacts_and_has_no_gps(tmp_path):
    source = Path(".sim/PX4-Autopilot")
    if not source.exists():
        pytest.skip("Pinned PX4 source checkout required")
    metadata = generate(Config(case="occluded"), source, tmp_path)
    drone = ET.parse(tmp_path / "world.sdf").find("world/model[@name='drone']")
    assert len(drone.findall("plugin[@name='gz::sim::systems::MulticopterMotorModel']")) == 4
    assert len(drone.findall("link/sensor[@type='rgbd_camera']")) == 6
    assert len(drone.findall("link/sensor[@type='contact']")) == 5
    assert len(drone.findall("link/sensor[@type='contact']/contact/topic")) == 5
    assert not drone.findall("link/sensor[@type='navsat']")
    assert any(box["name"] == "search_screen" for box in metadata["geometry"])


def test_dynamic_scene_records_reproducible_motion_recipe(tmp_path):
    source = Path(".sim/PX4-Autopilot")
    if not source.exists():
        pytest.skip("Pinned PX4 source checkout required")
    metadata = generate(Config(case="crossing", seed=123), source, tmp_path)
    actor = metadata["actors"][0]
    assert metadata["geometry"][actor["body"]]["center"][1] == pytest.approx(
        2 + 1.35 * np.sin(actor["phase"])
    )


def test_action_facts_do_not_filter_or_override_jev():
    from experiments.legacy.native_policy import JevExperiment

    class Gateway:
        def choose(self, state, instructions, criteria):
            assert set(criteria) == {"brake", "cruise_north"}
            assert state["movement_checks"]["cruise_north"]["violates_current_rules"]
            # Deliberately choose an invalid turn: the request layer must not
            # silently replace it with a safer algorithm-selected command.
            return {"latency_seconds": 0.1, "usage": {}, "choice": "cruise_north"}

    experiment = object.__new__(JevExperiment)
    experiment.config = SimpleNamespace(action_facts=True)
    experiment.world = SimpleNamespace(time=1.0, realtime=True)
    experiment.gateway = Gateway()
    experiment.cost, experiment.calls, experiment.error_streak = 0, [], 0
    choice, _ = experiment.request(
        "control",
        {"sensors": {"velocity_estimate": [0.5, 0, 0], "clearance_m": {"north": 3.5}}},
        "",
        {"brake": "Stop", "cruise_north": "Move north"},
    )
    assert choice == "cruise_north"


def test_motion_predictor_uses_the_x500_envelope():
    from jev_drone.perception.motion import MotionTracker

    risks = []
    for radius in (0.30, 0.40):
        tracker = MotionTracker(body_radius=radius)
        for index in range(4):
            tracker.update(
                index * 0.1,
                [
                    {
                        "color_bin": 2,
                        "position": np.array([1.4 - index * 0.1, 0.58, 1.3]),
                        "half_extent": np.array([0.2, 0.2, 0.2]),
                        "pixels": 50,
                    }
                ],
            )
        risks.append(
            tracker.observation(np.array([0.0, 0.0, 1.3]), np.zeros(3), 0.5, 0.3)[
                "current_course_risk"
            ]
        )
    assert risks == [False, True]


def test_flight_expires_control_before_accepting_or_snapshotting_a_patch(monkeypatch):
    import threading

    from jev_drone.control.controls import ControlLatch
    from jev_drone.sim import flight

    transport = object.__new__(flight.Flight)
    transport.lock = threading.Lock()
    transport.controls = ControlLatch()
    transport.target_position = None
    transport.expirations = []
    transport.clock = lambda: 20.0
    transport.position_at = 10.7
    transport.controls.update({"x": 0.5, "z": 0.25}, 10.0)
    pending = transport.controls.revision
    monkeypatch.setattr(flight.time, "monotonic", lambda: 10.7)
    receipt = transport.patch_velocity({"x": 0.0}, pending)
    assert not receipt["applied"]
    assert transport.control_snapshot()["velocity_mps"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert len(transport.expirations) == 1
    assert transport.expirations[0]["reason"] == "lease_expired"
