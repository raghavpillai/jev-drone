"""Behavioral contracts for real pan and front-only semantic observations."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jev_drone.control.controls import ControlLatch
from jev_drone.control.policy import CHANNELS, movement_reasons, patches


def test_pan_and_translation_are_independent_persistent_channels():
    controls = ControlLatch(axes=CHANNELS, state_key="body_controls")
    controls.update({"forward_mps": 0.25, "yaw_rate_rps": 0.6}, 0.0)
    controls.update(patches(0.5)["stop_forward"], 0.1, controls.revision)
    assert controls.snapshot()["body_controls"] == {
        "forward_mps": 0.0,
        "right_mps": 0.0,
        "up_mps": 0.0,
        "yaw_rate_rps": 0.6,
    }
    controls.update(patches(0.5)["backward_slow"], 0.2, controls.revision)
    controls.update(patches(0.5)["stop_pan"], 0.3, controls.revision)
    assert controls.values == {
        "forward_mps": -0.25,
        "right_mps": 0.0,
        "up_mps": 0.0,
        "yaw_rate_rps": 0.0,
    }
    controls.expire(0.91)
    assert all(v == 0 for v in controls.values.values())


def test_keeping_a_turn_does_not_hide_a_fast_translation_violation():
    observation = {
        "body_clearance_m": dict(forward=3.5, backward=None, up=2.0, down=1.0),
        "body_velocity_mps": dict(forward=0.4, up=0.0),
        "yaw_rate_rps": 0.4,
    }
    assert movement_reasons(dict(forward_mps=0.5, up_mps=0.0, yaw_rate_rps=0.0), observation) == [
        "cruise_while_turning"
    ]
    reasons = movement_reasons(dict(forward_mps=-0.25, up_mps=0.0, yaw_rate_rps=0.0), observation)
    assert "insufficient_or_unknown_backward_depth" in reasons
    assert "must_brake_before_reversing_forward" in reasons


def test_default_rig_covers_strafe_and_keeps_wide_front_view(tmp_path):
    from jev_drone.sim.config import Config
    from jev_drone.sim.scene import HFOV, generate

    source = Path(".sim/PX4-Autopilot")
    if not source.exists():
        pytest.skip("Pinned simulator source required")
    metadata = generate(Config(front_controls=True), source, tmp_path)
    assert set(metadata["cameras"]) == {"east", "west", "north", "south", "up", "down"}
    cameras = ET.parse(tmp_path / "world.sdf").findall(
        "world/model[@name='drone']/link/sensor[@type='rgbd_camera']"
    )
    assert len(cameras) == 6
    assert all(float(c.findtext("camera/horizontal_fov")) == pytest.approx(HFOV) for c in cameras)


def test_rear_pixels_cannot_reveal_targets_even_when_front_is_missing():
    pytest.importorskip("pymavlink")
    from jev_drone.perception.sensors import Frame
    from jev_drone.sim.backend import CameraSensors
    from jev_drone.sim.config import Config

    front_rgb, rear_rgb = np.zeros((24, 32, 3), np.uint8), np.zeros((24, 32, 3), np.uint8)
    front_rgb[5:18, 10:22, 0] = 255
    rear_rgb[5:18, 10:22, 2] = 255
    front = Frame(
        0.8,
        front_rgb,
        np.full((24, 32), 3.0),
        np.array([0.3, 0.0, 1.4]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        32 / (2 * math.tan(1.2)),
        "east",
    )
    rear = Frame(
        0.8,
        rear_rgb,
        np.full((24, 32), 3.0),
        np.array([-0.3, 0.0, 1.4]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        32 / (2 * math.tan(1.2)),
        "west",
    )
    sensor = object.__new__(CameraSensors)
    sensor.config = Config(front_controls=True)
    sensor.world = SimpleNamespace(
        time=1.0,
        pose_estimate=lambda: np.array([0.0, 0.0, 1.3]),
        camera_direction=np.array([1.0, 0.0, 0.0]),
        camera_target=np.array([1.0, 0.0, 0.0]),
        flight=SimpleNamespace(velocity=np.zeros(3), yaw_rate=0.0),
    )
    sensor.tracker = None
    sensor._frames = lambda: [front, rear]
    observation = sensor.observation()
    assert {d["marker"] for d in observation["detections"]} == {"red"}
    assert observation["body_clearance_m"]["backward"] is not None
    sensor._frames = lambda: [rear]
    observation = sensor.observation()
    assert observation["detections"] == [] and not observation["front_frame_available"]
    assert observation["body_clearance_m"]["forward"] is None


def test_body_velocity_tracks_heading_without_pitching_vertical_velocity():
    pytest.importorskip("pymavlink")
    from jev_drone.sim.flight import body_velocity_enu

    controls = dict(forward_mps=0.5, up_mps=0.25, yaw_rate_rps=0.6)
    assert np.allclose(body_velocity_enu(controls, math.pi / 2), [0.0, 0.5, 0.25])
    controls["forward_mps"] = -0.5
    assert np.allclose(body_velocity_enu(controls, math.pi / 2), [0.0, -0.5, 0.25])


@pytest.mark.parametrize("yaw_rate", [-0.6, 0.6])
def test_native_wire_uses_body_frame_and_correct_yaw_and_vertical_signs(yaw_rate):
    pytest.importorskip("pymavlink")
    import threading
    import time

    from jev_drone.sim.flight import Flight

    flight = object.__new__(Flight)
    flight.lock = threading.Lock()
    flight.body_controls = True
    flight.controls = ControlLatch(axes=CHANNELS, state_key="body_controls")
    flight.controls.update(
        dict(forward_mps=0.25, right_mps=0.2, up_mps=0.1, yaw_rate_rps=yaw_rate), time.monotonic()
    )
    flight.active = flight.running = True
    flight.target_position = None
    flight.position_at = time.monotonic()
    flight.clock = lambda: 42.0
    flight.rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    flight.expirations, flight.sent, packets = [], [], []

    def publish(*args):
        packets.append(args)
        flight.running = False

    flight.connection = SimpleNamespace(
        target_system=1,
        recv_match=lambda **kwargs: None,
        mav=SimpleNamespace(
            heartbeat_send=lambda *args: None, set_position_target_local_ned_send=publish
        ),
    )
    flight._loop()
    assert packets[0][3:5] == (8, 1479)
    assert np.allclose(packets[0][8:11], [0.25, 0.2, -0.1])
    assert packets[0][15] == -yaw_rate
    assert np.allclose(flight.sent[0]["velocity"], [0.2, 0.25, 0.1])


def test_signed_strafe_replaces_opposite_direction_without_changing_pan():
    controls = ControlLatch(axes=CHANNELS, state_key="body_controls")
    controls.update({"right_mps": -0.25, "yaw_rate_rps": 0.2}, 0.0)
    controls.update({"right_mps": 0.25}, 0.1, controls.revision)
    assert controls.values["right_mps"] == 0.25
    assert controls.values["yaw_rate_rps"] == 0.2
    controls.update(patches(0.5)["stop_strafe"], 0.2, controls.revision)
    assert controls.values["right_mps"] == 0.0
    assert controls.values["yaw_rate_rps"] == 0.2


def test_diagonal_motion_needs_observed_diagonal_clearance():
    observation = {
        "body_clearance_m": dict(forward=3.5, right=3.5),
        "body_velocity_mps": dict(forward=0.0, right=0.0),
        "yaw_rate_rps": 0.0,
    }
    values = dict(forward_mps=0.25, right_mps=0.25, up_mps=0.0, yaw_rate_rps=0.0)
    assert movement_reasons(values, observation) == ["insufficient_or_unknown_combined_depth"]
    assert movement_reasons(values, observation, 0.9) == ["insufficient_or_unknown_combined_depth"]
    assert movement_reasons(values, observation, 1.3) == []


def test_fused_depth_supports_diagonals_and_keeps_blind_regions_unknown():
    from jev_drone.perception.depth import clearance
    from jev_drone.perception.sensors import Frame

    pose = np.array([0.0, 0.0, 1.3])

    def frame(angle, distance):
        forward = np.array([math.cos(angle), math.sin(angle), 0.0])
        right = np.array([math.sin(angle), -math.cos(angle), 0.0])
        return Frame(
            0.0,
            np.zeros((120, 160, 3)),
            np.full((120, 160), distance),
            pose,
            forward,
            right,
            np.array([0.0, 0.0, 1.0]),
            160 / (2 * math.tan(1.2)),
        )

    frames = [frame(0.0, 4.0), frame(math.pi / 2, 4.0)]
    assert clearance(frames, pose, np.array([1.0, 1.0, 0.0])) >= 1.3
    assert clearance([], pose, np.array([1.0, 1.0, 0.0])) is None
    assert clearance([frame(0.0, 0.3)], pose, np.array([1.0, 0.0, 0.0])) == 0.0


def test_progress_facts_cannot_filter_or_change_a_movement():
    from jev_drone.control.progress_facts import enrich

    state = {
        "sensors": {"body_clearance_m": dict(forward=0.0, right=2.5, left=None)},
        "current_controls": {"body_controls": dict.fromkeys(CHANNELS, 0.0)},
        "at_position": False,
        "control_effects": {
            "keep_controls": {"patch": {}},
            "detour_right": {"patch": dict.fromkeys(CHANNELS, 0.0)},
            "detour_left": {"patch": dict.fromkeys(CHANNELS, 0.0)},
            "forward_slow": {"patch": {"forward_mps": 0.25}},
        },
    }
    criteria = dict.fromkeys(state["control_effects"], "Original option.")
    enriched, described = enrich(state, criteria)
    assert set(described) == set(criteria)
    for name in criteria:
        assert enriched["control_effects"][name]["patch"] == state["control_effects"][name]["patch"]
    assert enriched["control_effects"]["detour_right"]["bypass_visibility"] == "OPEN"
    assert enriched["control_effects"]["detour_left"]["bypass_visibility"] == "UNKNOWN"
    assert "progress" not in state["control_effects"]["keep_controls"]
    reached, described = enrich(
        {**state, "at_position": True, "goal_path_clearance_m": 0.0}, criteria
    )
    assert reached["goal_path_status"] == "ARRIVED"
    assert "ALREADY REACHED" in described["detour_right"]


def test_vertical_transition_releases_forward_and_repetition_is_empty():
    from jev_drone.control.controls import changed_channels

    controls = ControlLatch(axes=CHANNELS, state_key="body_controls")
    controls.update({"forward_mps": 0.25}, 0.0)
    desired = patches(0.5)["down"]
    patch = changed_channels(controls.values, desired)
    assert patch == {"forward_mps": 0.0, "up_mps": -0.25}
    controls.update(patch, 0.1, controls.revision)
    assert controls.values == desired
    patch = changed_channels(controls.values, desired)
    assert patch == {}
    controls.update(patch, 0.5, controls.revision)
    assert controls.expire(0.9) is None
    assert controls.expire(1.11)["after"]["body_controls"] == dict.fromkeys(CHANNELS, 0.0)


def test_panning_near_an_observed_obstacle_requires_translation_away_first():
    observation = {
        "nearest_observed_body_gap_m": -0.01,
        "body_clearance_m": {"backward": 2.5},
        "body_velocity_mps": {"forward": 0.0},
        "yaw_rate_rps": 0.0,
    }
    assert movement_reasons(patches(0.5)["pan_left"], observation) == [
        "rotation_close_to_observed_obstacle"
    ]
    assert movement_reasons(patches(0.5)["backward_slow"], observation) == []


def test_altitude_goal_is_not_satisfied_by_the_old_horizontal_radius():
    from jev_drone.control.policy import FrontExperiment
    from jev_drone.sim.config import Config

    experiment = object.__new__(FrontExperiment)
    experiment.config = Config()
    assert not experiment.at_goal(np.array([0.0, 0.0, 0.30]))
    assert experiment.at_goal(np.array([0.30, 0.0, 0.10]))


def test_short_bypass_can_fit_below_a_ceiling_when_full_step_cannot():
    from jev_drone.control.progress_facts import enrich

    state = {
        "sensors": {"body_clearance_m": {"up": 0.9}},
        "current_controls": {"body_controls": dict.fromkeys(CHANNELS, 0.0)},
        "at_position": False,
        "control_effects": {"detour_up": {}, "detour_medium_up": {}, "detour_short_up": {}},
    }
    enriched, _ = enrich(state, dict.fromkeys(state["control_effects"], "Select a bypass."))
    assert not enriched["control_effects"]["detour_up"]["bypass_viable"]
    assert enriched["control_effects"]["detour_short_up"]["bypass_viable"]
    assert enriched["control_effects"]["detour_medium_up"]["bypass_viable"]
    assert enriched["control_effects"]["detour_medium_up"]["bypass_direction"] == "up"
    assert enriched["control_effects"]["detour_medium_up"]["bypass_distance_m"] == 0.9


def test_target_approaches_allow_front_view_instead_of_hovering_above_item():
    from jev_drone.control.policy import FrontExperiment
    from jev_drone.sim.config import Config

    experiment = object.__new__(FrontExperiment)
    experiment.config = Config()
    experiment.stage = 0
    experiment.world = SimpleNamespace(mission=[{"marker": "red", "room": "bedroom"}])
    experiment.memory = {"red": {"position": [10.7, 2.8, 1.5]}}
    experiment.sensors = SimpleNamespace(observed_gap_at=lambda point: 0.3, current=[])
    observation = {
        "position_estimate": [10.6, 2.8, 3.1],
        "room": "bedroom",
        "speed": 0.0,
        "yaw_rate_rps": 0.0,
        "detections": [],
    }
    approaches = {
        k: v for k, v in experiment.options(observation).items() if k.startswith("approach_marker")
    }
    assert "approach_marker_south" in approaches and "approach_marker_west" in approaches
    assert all(
        a["position"][2] == 1.5 and a["look_position"] == [10.7, 2.8, 1.5]
        for a in approaches.values()
    )
    assert all(
        6.45 <= a["position"][0] <= 11.55 and 0.45 <= a["position"][1] <= 3.55
        for a in approaches.values()
    )


def handoff_experiment():
    from jev_drone.control.policy import FrontExperiment
    from jev_drone.sim.config import Config

    experiment = object.__new__(FrontExperiment)
    experiment.config = Config()
    experiment.world = SimpleNamespace(mission=[{"marker": "red", "room": "bedroom"}], time=10.0)
    experiment.stage = 0
    experiment.target_notified_stage = None
    experiment.task = {"name": "view_bedroom_0_0_1.3", "position": [7.5, 1.0, 1.3]}
    experiment.tasks = [{**experiment.task, "outcome": "active"}]
    experiment.detour = {"position": [10.0, 3.0, 1.3]}
    observation = {
        "room": "bedroom",
        "position_estimate": [9.5, 2.8, 1.5],
        "speed": 0.0,
        "yaw_rate_rps": 0.0,
        "detections": [{"marker": "red", "position": [10.7, 2.8, 1.5], "pixels": 30}],
    }
    return experiment, observation


def test_ready_inspection_interrupts_waypoint_without_reporting_or_advancing_stage():
    experiment, observation = handoff_experiment()
    assert experiment.task_finished(observation)
    assert experiment.tasks[-1]["outcome"] == "objective_ready"
    assert experiment.task is None and experiment.detour is None
    assert experiment.stage == 0


def test_new_target_interrupts_search_once_but_moving_is_not_inspection_ready():
    experiment, observation = handoff_experiment()
    observation["speed"] = 0.3
    assert experiment.mission_event(observation) == "target_observed"
    assert experiment.mission_event(observation) is None
    observation["speed"] = 0.0
    observation["detections"] = []
    assert experiment.mission_event(observation) is None


def test_maneuver_course_does_not_use_mission_inspection_events():
    from jev_drone.sim.controller_check import ControllerCheck

    experiment = object.__new__(ControllerCheck)
    # Its course stages do not index the room-search mission at all.
    assert experiment.mission_event({}) is None


def test_neutral_but_coasting_is_braking_not_a_stalled_hover():
    from jev_drone.control.decision_instructions import braking_facts

    state = {
        "current_controls": {"body_controls": dict.fromkeys(CHANNELS, 0.0)},
        "sensors": {"speed": 0.25, "yaw_rate_rps": 0.0},
        "at_position": False,
        "control_effects": {
            "keep_controls": {"patch": {}, "movement_violations": []},
            "up": {
                "patch": {"up_mps": 0.25},
                "movement_violations": ["must_brake_before_reversing_up"],
            },
        },
    }
    criteria = dict.fromkeys(state["control_effects"], "Original.")
    enriched, described = braking_facts(state, criteria)
    assert enriched["neutral_controls_still_braking"]
    assert not enriched["holding_position_away_from_goal"]
    assert enriched["control_effects"]["keep_controls"]["progress"] == "braking_measured_motion"
    assert described["up"].startswith("DO NOT SELECT")
    assert set(described) == set(criteria)
    assert enriched["control_effects"]["up"] == state["control_effects"]["up"]
    assert "progress" not in state["control_effects"]["keep_controls"]


def test_keeping_an_unsafe_nonzero_control_is_still_marked_unsafe():
    from jev_drone.control.decision_instructions import braking_facts

    state = {
        "current_controls": {
            "body_controls": {**dict.fromkeys(CHANNELS, 0.0), "forward_mps": 0.25}
        },
        "sensors": {"speed": 0.25, "yaw_rate_rps": 0.0},
        "at_position": False,
        "control_effects": {
            "keep_controls": {
                "patch": {},
                "movement_violations": ["must_brake_before_reversing_forward"],
            }
        },
    }
    enriched, described = braking_facts(state, {"keep_controls": "Keep."})
    assert not enriched["neutral_controls_still_braking"]
    assert described["keep_controls"].startswith("DO NOT SELECT")


@pytest.mark.parametrize(
    "goal,room,pose,complete",
    [
        ([7.0, 2.0, 1.3], "bedroom", [7.6, 2.5, 1.5], True),
        ([7.0, 2.0, 1.3], "bedroom", [6.3, 2.0, 1.3], False),
        ([5.0, 2.0, 1.3], "living", [5.3, 2.0, 1.3], True),
        ([5.0, 2.0, 1.3], "bedroom", [7.0, 2.0, 1.3], False),
    ],
)
def test_crossing_finishes_beyond_door_body_margin_not_at_exact_waypoint(
    goal, room, pose, complete
):
    from jev_drone.control.policy import FrontExperiment

    experiment = object.__new__(FrontExperiment)
    experiment.task = {"name": "cross_living_bedroom", "position": goal}
    assert experiment.crossed_doorway({"room": room, "position_estimate": pose}) == complete


@pytest.mark.parametrize(
    "objective", [{"dock": [1.4, 2.0, 1.3]}, {"marker": "red", "room": "bedroom"}]
)
def test_travel_planner_has_intermediate_goals_available_to_transit_mode(objective):
    from jev_drone.control.policy import FrontExperiment
    from jev_drone.planning.return_memory import ReturnMemory
    from jev_drone.planning.search_memory import SearchMemory
    from jev_drone.sim.config import Config

    experiment = object.__new__(FrontExperiment)
    experiment.config = Config()
    experiment.stage = 0
    experiment.world = SimpleNamespace(mission=[objective])
    experiment.memory = {}
    experiment.search_memory = SearchMemory()
    experiment.return_memory = ReturnMemory()
    experiment.tasks = []
    experiment.sensors = SimpleNamespace(current=[], observed_gap_at=lambda point: None)
    observation = {
        "room": "living",
        "position_estimate": [3.9, 6.5, 1.3],
        "speed": 0.0,
        "yaw_rate_rps": 0.0,
        "detections": [],
    }
    options = experiment.options(observation)
    assert options["transit_living_0_1_1.3"]["position"] == [1.5, 7.0, 1.3]
    assert options["transit_living_0_1_1.3"]["observed_path_clearance_m"] is None
    if "dock" in objective:
        assert not any(n.startswith("view_") for n in options)
    assert ("return_launch" in options) == ("dock" in objective)


def test_search_endpoint_on_observed_furniture_is_flagged_without_removing_it():
    pytest.importorskip("pymavlink")
    from jev_drone.control.policy import FrontExperiment
    from jev_drone.sim.backend import CameraSensors
    from jev_drone.sim.config import Config

    experiment = object.__new__(FrontExperiment)
    experiment.config = Config()
    experiment.stage = 0
    experiment.world = SimpleNamespace(mission=[{"marker": "red", "room": "bedroom"}])
    experiment.memory = {}
    experiment.sensors = object.__new__(CameraSensors)
    experiment.sensors.current = []
    experiment.sensors.observed_cloud = np.array([[10.4, 1.0, 0.9]])
    observation = {
        "room": "bedroom",
        "position_estimate": [7.5, 1.0, 1.3],
        "speed": 0.0,
        "yaw_rate_rps": 0.0,
        "detections": [],
    }
    options = experiment.options(observation)
    assert options["view_bedroom_1_0_1.3"]["observed_endpoint_unsafe"]
    assert not options["view_bedroom_0_0_1.3"]["observed_endpoint_unsafe"]
