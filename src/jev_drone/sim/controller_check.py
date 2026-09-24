"""Known local goals; every flight command is selected by live Jev."""
import math
import numpy as np

from jev_drone.control.policy import FrontExperiment, wrap


class ControllerCheck(FrontExperiment):
    mission_handoffs = False

    course = [
        ("forward", [2.4, 2., 1.3], 0.),
        ("backward", [1.4, 2., 1.3], 0.),
        ("strafe_left", [1.4, 3., 1.3], 0.),
        ("strafe_right", [1.4, 2., 1.3], 0.),
        ("pan_left", [1.4, 2., 1.3], 90.),
        ("strafe_right_after_turn", [2.4, 2., 1.3], 90.),
        ("strafe_left_after_turn", [1.4, 2., 1.3], 90.),
        ("pan_right", [1.4, 2., 1.3], 0.),
        ("up", [1.4, 2., 2.3], 0.),
        ("down", [1.4, 2., 1.3], 0.),
    ]

    def plan(self, observation):
        if self.tasks:
            task = self.tasks[-1]
            error = abs(wrap(self.world.transport.truth_yaw-math.radians(task["heading_degrees"])))
            valid = (task["outcome"] == "arrived"
                     and np.linalg.norm(self.world.position-task["position"]) <= .48
                     and error <= math.radians(15) and np.linalg.norm(self.world.velocity) <= .15)
            self.reports.append({"task": task["name"], "valid": bool(valid), "time": self.world.time,
                                 "position": self.world.position.tolist(), "truth_yaw_degrees": math.degrees(self.world.transport.truth_yaw)})
            if not valid:
                self.world.status = "controller_check_failed"
                return
            self.stage += 1
        if self.stage == len(self.course):
            self.world.status = "success"
            return
        name, position, heading = self.course[self.stage]
        self.task = dict(name=name, position=position, heading_degrees=heading,
                         purpose="Reach the position while maintaining the requested camera heading; strafe rather than yaw to move sideways.")
        self.task_started = self.last_progress = self.world.time
        self.best_distance = self.distance(observation)
        self.best_heading_error = math.inf
        self.detour = None
        self.tasks.append({**self.task, "time": self.world.time, "start": observation["position_estimate"], "outcome": "active"})

    def run(self):
        result = super().run()
        result["evaluation"] = "Jev local controller qualification; supplied test goals, no model mission planner"
        result["course"] = self.course
        return result
