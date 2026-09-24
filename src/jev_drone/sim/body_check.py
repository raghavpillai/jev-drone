"""Scripted native check of yaw, heading-relative translation and partial release."""
import math
import json
import time

import numpy as np

from jev_drone.control.controls import BODY_CHANNELS


def run(world, sensors, output):
    phases, receipts = [], []
    def yaw():
        return math.atan2(world.camera_direction[1], world.camera_direction[0])
    def update(patch, revision=None):
        revision = world.flight.control_snapshot()["revision"] if revision is None else revision
        receipt = world.update_controls("scripted_body_check", patch, revision)
        receipts.append(receipt)
        return receipt
    def phase(name, patch, seconds):
        position, heading, started, sim_start = world.position.copy(), yaw(), time.monotonic(), world.time
        desired = {**world.flight.control_snapshot()["body_controls"], **patch}
        update(patch)
        while world.time-sim_start < seconds:
            assert world.status == "running", world.status
            assert time.monotonic()-started < seconds*4+5, "Calibration physics stalled"
            # A simulated telemetry pause expires the latch. Resume the explicitly
            # scripted phase only after telemetry recovers, never renew blind motion.
            if world.telemetry_ready:
                current = world.flight.control_snapshot()["body_controls"]
                update({key: value for key, value in desired.items() if current[key] != value})
            time.sleep(.1)
        sensors.observation()
        sensors.save_frame(output/(name+".npz"))
        displacement = world.position-position
        summary = {"name": name, "time": world.time, "sim_seconds": world.time-sim_start,
                   "wall_seconds": time.monotonic()-started, "displacement": displacement.tolist(),
                   "along_initial_heading": float(displacement@np.array([math.cos(heading), math.sin(heading), 0.])),
                   "along_initial_right": float(displacement@np.array([math.sin(heading), -math.cos(heading), 0.])),
                   "yaw_change_degrees": math.degrees(math.atan2(math.sin(yaw()-heading), math.cos(yaw()-heading))),
                   "body_controls": world.flight.control_snapshot()["body_controls"]}
        phases.append(summary)
        (output/"body-check-phases.json").write_text(json.dumps(phases, indent=2))
        return summary
    stop = dict.fromkeys(BODY_CHANNELS, 0.)
    sensors.observation()
    sensors.save_frame(output/"initial-frame.npz")
    left = phase("pan_left", {"yaw_rate_rps": .6}, math.pi/1.2)
    phase("stop_left", {"yaw_rate_rps": 0.}, 2.)
    forward = phase("forward_after_turn", {"forward_mps": .25}, 2.5)
    phase("stop_forward", {"forward_mps": 0.}, 2.)
    backward = phase("backward_after_turn", {"forward_mps": -.25}, 2.5)
    phase("stop_backward", {"forward_mps": 0.}, 2.)
    strafe_right = phase("strafe_right_after_turn", {"right_mps": .25}, 2.5)
    phase("stop_strafe_right", {"right_mps": 0.}, 2.)
    strafe_left = phase("strafe_left_after_turn", {"right_mps": -.25}, 2.5)
    phase("stop_strafe_left", {"right_mps": 0.}, 2.)
    assert strafe_right["along_initial_right"] > .35 and strafe_left["along_initial_right"] < -.35
    assert abs(strafe_right["yaw_change_degrees"]) < 3 and abs(strafe_left["yaw_change_degrees"]) < 3
    right = phase("pan_right", {"yaw_rate_rps": -.6}, math.pi/1.2)
    phase("stop_right", {"yaw_rate_rps": 0.}, 2.)
    up = phase("up", {"up_mps": .25}, 2.)
    phase("stop_up", {"up_mps": 0.}, 2.)
    down = phase("down", {"up_mps": -.25}, 2.)
    phase("stop_down", {"up_mps": 0.}, 2.)
    assert up["displacement"][2] > .25 and down["displacement"][2] < -.25
    phase("forward_and_pan", {"forward_mps": .2, "yaw_rate_rps": .2}, 1.5)
    released = phase("release_forward_keep_pan", {"forward_mps": 0.}, 1.)
    assert released["body_controls"]["yaw_rate_rps"] == .2
    phase("stop_all", stop, 2.)
    update({"forward_mps": .15, "yaw_rate_rps": -.2})
    pending = world.flight.control_snapshot()["revision"]
    time.sleep(world.lease+.25)
    assert not update({"forward_mps": 0.}, pending)["applied"]
    phase("after_expiry", {}, 3.)
    assert left["yaw_change_degrees"] > 60 and right["yaw_change_degrees"] < -60
    assert forward["along_initial_heading"] > .35 and backward["along_initial_heading"] < -.35
    assert np.linalg.norm(world.velocity) < .05 and abs(world.flight.yaw_rate) < .05
    assert not world.contacts
    return {"status": "body_controls_verified", "world": world.metadata(), "phases": phases, "receipts": receipts,
            "trace": world.trace, "contacts": world.contacts, "sim_seconds": world.time,
            "wall_seconds": time.monotonic()-world.started_wall, "end_speed": float(np.linalg.norm(world.velocity)),
            "telemetry_outages": world.telemetry_outages,
            "end_yaw_rate_rps": world.flight.yaw_rate}
