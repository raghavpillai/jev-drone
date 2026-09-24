"""Persistence, zero release, watchdog races, and matched experimental actions."""

from types import SimpleNamespace

import pytest

from experiments.legacy.native_policy import JevExperiment, control_options
from jev_drone.control.controls import ControlLatch


def test_partial_release_preserves_other_channels_and_retries_are_idempotent():
    latch = ControlLatch()
    latch.update({"x": 0.5, "y": 0.25, "z": -0.25}, 0.0)
    latch.update({"x": 0.0}, 0.1, latch.revision)
    assert latch.values == {"x": 0.0, "y": 0.25, "z": -0.25}
    revision = latch.revision
    latch.update({"y": 0.5}, 0.2, revision)
    replay = latch.update({"y": 0.5}, 0.3, revision)
    assert not replay["applied"]
    assert latch.values == {"x": 0.0, "y": 0.5, "z": -0.25}
    latch.update({"y": 0.5}, 0.3, latch.revision)
    assert latch.values["y"] == 0.5


def test_keep_is_a_heartbeat_but_silence_stops_and_invalidates_pending_patch():
    latch = ControlLatch()
    latch.update({"x": 0.5}, 0.0)
    latch.update({}, 0.5, latch.revision)
    assert latch.expire(1.09) is None
    before = latch.revision
    event = latch.expire(1.1)
    assert event["reason"] == "lease_expired"
    assert latch.values == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert not latch.update({"y": 0.25}, 1.11, before)["applied"]
    assert not latch.update({}, 1.12, before)["applied"]
    assert latch.expire(2.0) is None
    latch.update({}, 2.0, latch.revision)
    assert all(v == 0 for v in latch.values.values())


def test_explicit_stop_or_lost_telemetry_invalidates_inflight_update():
    latch = ControlLatch()
    latch.update({"x": 0.5}, 0.0)
    revision = latch.revision
    latch.update({"x": 0.0, "y": 0.0, "z": 0.0}, 0.1)
    assert not latch.update({"x": 0.5}, 0.2, revision)["applied"]
    latch.update({"x": 0.5}, 0.3)
    revision = latch.revision
    assert latch.expire(0.31, telemetry_stale=True)["reason"] == "telemetry_stale"
    assert not latch.update({}, 0.32, revision)["applied"]


@pytest.mark.parametrize(
    "patch",
    [{"x": 0.25, "q": 0}, {"y": float("nan")}, {"z": float("inf")}, {"x": True}, {"y": 2.0}],
)
def test_invalid_update_cannot_partially_mutate_or_renew_controls(patch):
    latch = ControlLatch()
    latch.update({"x": 0.5}, 0.0)
    snapshot, expires = latch.snapshot(), latch.expires
    with pytest.raises(ValueError):
        latch.update(patch, 0.1, latch.revision)
    assert latch.snapshot() == snapshot and latch.expires == expires


@pytest.mark.parametrize(
    "current", [dict(x=0.0, y=0.0, z=0.0), dict(x=0.5, y=0.0, z=0.0), dict(x=0.0, y=-0.25, z=0.0)]
)
def test_full_and_patch_offer_identical_physical_actions(current):
    criteria = dict.fromkeys(
        [
            "brake",
            "slow_east",
            "cruise_east",
            "slow_south",
            "cruise_up",
            "look_north",
            "detour_west",
        ],
        "description",
    )
    snapshot = {"velocity_mps": current, "revision": 4}
    full = control_options(criteria, snapshot, 0.5, "full")
    patch = control_options(criteria, snapshot, 0.5, "patch")
    assert {v["action"] for v in full.values()} == {v["action"] for v in patch.values()}
    for option in patch.values():
        applied = {**current, **option["patch"]}
        assert applied == full[option["action"]]["resulting_velocity_mps"]
        assert all(current[k] != v for k, v in option["patch"].items())
    assert "keep_controls" in patch
    assert patch["keep_controls"]["patch"] == {}


def test_native_request_preserves_jev_patch_choice_and_exposes_keep_risk():
    class Gateway:
        def choose(self, state, instructions, criteria):
            assert state["movement_checks"]["keep_controls"]["violates_current_rules"]
            assert "keep_controls" in criteria  # evidence does not filter the choice
            return {"latency_seconds": 0.1, "usage": {}, "choice": "keep_controls", "state": state}

    experiment = object.__new__(JevExperiment)
    experiment.config = SimpleNamespace(action_facts=True, control_interface="patch", speed=0.5)
    experiment.world = SimpleNamespace(
        time=1.0,
        realtime=True,
        flight=SimpleNamespace(
            control_snapshot=lambda: {"velocity_mps": {"x": 0.5, "y": 0.0, "z": 0.0}, "revision": 2}
        ),
    )
    experiment.gateway = Gateway()
    experiment.cost, experiment.calls, experiment.error_streak = 0, [], 0
    choice, log = experiment.request(
        "control",
        {"sensors": {"velocity_estimate": [0.5, 0, 0], "clearance_m": {"east": 0.2}}},
        "",
        {"brake": "Stop", "cruise_east": "Move east"},
    )
    assert choice == "keep_controls" and log["effective_choice"] == "cruise_east"


def test_equal_slow_and_cruise_speed_retains_both_classification_choices():
    snapshot = {"velocity_mps": dict(x=0.25, y=0.0, z=0.0), "revision": 1}
    options = control_options(
        dict.fromkeys(("brake", "slow_east", "cruise_east")), snapshot, 0.25, "patch"
    )
    assert len(options) == 3
    assert {o["action"] for o in options.values()} == {"brake", "slow_east", "cruise_east"}


def test_rejected_patch_brakes_without_adopting_its_detour():
    experiment = object.__new__(JevExperiment)
    experiment.config = SimpleNamespace(control_interface="patch")
    brakes = []
    experiment.world = SimpleNamespace(
        update_controls=lambda *args: {"applied": False}, command=brakes.append
    )
    experiment.detour, experiment.stale, experiment.violations = None, 0, []
    log = {
        "control_options": {"detour_up": {"patch": {"x": 0.0}, "action": "detour_up"}},
        "state": {"current_controls": {"revision": 4}},
    }
    assert not experiment.apply_control("detour_up", {}, log)
    assert experiment.detour is None and experiment.stale == 1 and brakes == ["brake"]
    assert log["rejection_reason"] == "control_state_changed_during_request"
