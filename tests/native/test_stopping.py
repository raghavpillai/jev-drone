from jev_drone.control.controls import BODY_CHANNELS, ControlLatch
from jev_drone.control.stopping import required_clearance

LIMITS = dict(
    slow_clearance_m=0.7, full_clearance_m=1.5, command_lease_seconds=0.9, stopping_margin=True
)


def test_allowance_grows_with_measured_speed_sensor_age_and_lease():
    basic = required_clearance(0.25, 0.25, 0.1, LIMITS)
    assert basic == 0.7
    assert required_clearance(0.25, 0.4, 0.1, LIMITS) > basic
    assert required_clearance(0.25, 0.25, 0.65, LIMITS) > basic
    assert required_clearance(0.25, 0.25, 0.1, {**LIMITS, "command_lease_seconds": 1.2}) > basic


def test_longer_lease_still_stops_and_rejects_old_control_revision():
    latch = ControlLatch(0.9, axes=BODY_CHANNELS, state_key="body_controls")
    latch.update({"forward_mps": 0.25}, 0.0)
    revision = latch.revision
    assert latch.expire(0.7) is None
    assert latch.expire(0.91)["after"]["body_controls"]["forward_mps"] == 0.0
    assert not latch.update({"forward_mps": 0.25}, 0.92, revision)["applied"]
    latch.update({"forward_mps": 0.25}, 1.0)
    assert latch.expire(1.1, telemetry_stale=True)["reason"] == "telemetry_stale"


def test_settling_allowance_covers_native_slow_stop_counterexamples():
    limits = {**LIMITS, "slow_clearance_m": 0.0, "braking_settling_allowance_s": 0.7}
    # Sampled native stops exceeded v²/(2*0.5), before the reaction allowance.
    for speed, observed_displacement in ((0.160, 0.134), (0.261, 0.194)):
        total = required_clearance(speed, speed, 0.1, limits)
        reaction_and_padding = speed * (0.1 + 2 * 0.9) + 0.08
        assert total - reaction_and_padding >= observed_displacement
    assert required_clearance(0.25, 0.25, 0.1, limits) > required_clearance(0.25, 0.25, 0.1, LIMITS)
