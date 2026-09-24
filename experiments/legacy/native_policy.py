"""Matched full-command / partial-update interfaces; Jev selects every action."""
import json
import numpy as np

from jev_drone.planning.agent import Experiment
from jev_drone.world.world import DIRECTIONS


def velocity_for(choice, speed):
    if choice.startswith(("slow_", "cruise_")):
        pace, direction = choice.split("_", 1)
        return np.array(DIRECTIONS[direction])*(.25 if pace == "slow" else speed)
    return np.zeros(3)


def control_options(criteria, snapshot, speed, interface):
    """Same reachable vectors in both arms, expressed as replacements or patches."""
    options = {}
    for action in criteria:
        values = dict(zip(("x", "y", "z"), velocity_for(action, speed).tolist()))
        patch = {k: v for k, v in values.items() if v != snapshot["velocity_mps"][k]}
        name = action
        if interface == "patch" and not action.startswith(("look_", "detour_")):
            name = "__".join(f"set_{k}_{str(v).replace('-', 'minus').replace('.', 'p')}" for k, v in patch.items()) or "keep_controls"
        if name in options:
            # At cruise=.25, slow and cruise have equal effects but remain
            # distinct choices with their original clearance requirements.
            name += "__"+action
        options[name] = {"action": action, "patch": patch if interface == "patch" else values,
                         "resulting_velocity_mps": values}
    return options


class JevExperiment(Experiment):
    def request(self, role, state, instructions, criteria):
        if role == "control" and self.config.action_facts:
            observation = state["sensors"]
            velocity = np.array(observation["velocity_estimate"])
            speed = np.linalg.norm(velocity)
            criteria = dict(criteria)
            checks = {}
            for choice in criteria:
                if not choice.startswith(("slow_", "cruise_")):
                    continue
                pace, direction = choice.split("_", 1)
                depth = observation["clearance_m"][direction]
                reasons = []
                if depth is None or depth < (.4 if pace == "slow" else 1.3):
                    reasons.append("insufficient_or_unknown_camera_clearance")
                if speed > .15 and np.dot(velocity/speed, DIRECTIONS[direction]) < .95:
                    reasons.append("must_brake_before_this_axis_change")
                if observation.get("predicted_risk_by_direction", {}).get(direction):
                    reasons.append("predicted_moving_obstacle_risk")
                checks[choice] = {"violates_current_rules": bool(reasons), "reasons": reasons}
                criteria[choice] += (" CURRENTLY PROHIBITED BY THE RULES: "+", ".join(reasons)+". Select brake first."
                                     if reasons else " Passes the current observed movement checks; still select only if it advances your chosen goal.")
            state = {**state, "movement_checks": checks}
            instructions += "\nRead movement_checks before choosing. A movement with violates_current_rules=true is prohibited now. Select brake to stop before an axis change; do not combine braking and the new movement into one choice. These checks describe sensor evidence, not guaranteed future safety. All choices remain available and YOU are responsible for the selection.\n"
        options = None
        interface = getattr(self.config, "control_interface", None)
        if role == "control" and interface:
            snapshot = self.world.flight.control_snapshot()
            options = control_options(criteria, snapshot, self.config.speed, interface)
            state = {**state, "current_controls": snapshot,
                     "control_interface": interface,
                     "control_effects": {name: {k: v for k, v in option.items() if k != "action"}
                                         for name, option in options.items()}}
            if "movement_checks" in state:
                state["movement_checks"] = {name: state["movement_checks"][option["action"]]
                    for name, option in options.items() if option["action"] in state["movement_checks"]}
            criteria = {name: json.dumps(option["patch"])+" -> "+criteria[option["action"]]
                        for name, option in options.items()}
            instructions += "\nNative flight uses six continuously active body cameras. Look selects attention while braking; it does not rotate the drone. x=east, y=north, z=up. Controls are velocity setpoints in m/s, not motor throttle. Current controls and measured velocity differ because braking takes time.\n"
            if interface == "patch":
                instructions += """\nChoose an UPDATE to current_controls. Each choice's patch sets ONLY its listed
axes to absolute velocities; omitted axes keep their current values. Zero releases
that axis. keep_controls makes no change, but explicitly confirms it is still safe
to continue for another 0.6 seconds. Read control_effects for the resulting vector.
All earlier movement/braking rules apply to the RESULTING controls, including
keep_controls. To brake, select the update whose resulting velocity is all zero;
keeping a moving control is NOT braking. Updates are atomic; retries do not add
velocity. A response based on controls that expired meanwhile will be rejected.
"""
            else:
                instructions += "\nEach choice replaces all three velocity controls. Repeating a moving choice maintains it and confirms it is still safe for another 0.6 seconds.\n"
        choice, log = super().request(role, state, instructions, criteria)
        if options is not None:
            log["control_options"] = options
            if choice is not None:
                log["effective_choice"] = options[choice]["action"]
        return choice, log

    def apply_control(self, choice, observation, log):
        if self.config.control_interface != "patch":
            return super().apply_control(choice, observation, log)
        option = log["control_options"][choice]
        receipt = self.world.update_controls(choice, option["patch"],
                                             log["state"]["current_controls"]["revision"])
        log["control_receipt"] = receipt
        if not receipt["applied"]:
            self.stale += 1
            log["rejection_reason"] = "control_state_changed_during_request"
            self.world.command("brake")
            return False
        action = option["action"]
        self.audit(action, observation, log)
        if action.startswith("detour_"):
            self.select_detour(action, observation)
        elif action.startswith("look_"):
            self.world.camera_target = np.array(DIRECTIONS[action[5:]], dtype=float)
        return True
