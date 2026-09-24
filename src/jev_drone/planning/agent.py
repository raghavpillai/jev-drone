"""Jev chooses both persistent navigation tasks and every flight command."""
from collections import Counter
import json
import time

import numpy as np

from jev_drone.perception.sensors import Sensors
from jev_drone.world.world import DIRECTIONS, World
from jev_drone.world.layout import SMALL


PLANNER = """You are the mission planner for an indoor drone. Select ONE task.
Complete the ordered mission: visually inspect each requested colored marker in
its specified room, then return to launch and dock. Marker locations and furniture
are unknown until observed by cameras. Room boundaries and door positions are known.
Use door approach then door crossing to reach another room. Do not fly through walls.
In the requested room, look around from useful different positions and heights.
Remember unsuccessful views and stalled tasks; avoid repeating them unchanged.
An observed marker is a lead: approach its observed position, stop, face it, and
report inspection only when inspection_ready is true. A remembered location alone
does not prove the marker is still there. A task may need a detour around furniture.
At launch, dock only if dock_ready is true. Never declare success to save time.
You can give_up if sensing or navigation cannot support completion. All tasks are
selected by you; there is no route solver or obstacle-avoidance autopilot.
known_room_connections is the architectural room/door graph, not observed door
clearance. For a distant objective, follow a sequence of connected rooms: the
next room need not be the destination. Choose each next doorway yourself, keep
the same room leg through approach and crossing, and reassess after entering it.
Never aim directly through intervening walls at a distant object's coordinate.
Only work on ACTIVE_OBJECTIVE. Earlier mission stages are already finished.
If ACTIVE_OBJECTIVE is return_to_launch, stop all marker searching and navigate
through doors to the launch room and coordinate in ACTIVE_OBJECTIVE. Do not revisit inspected markers.
If a doorway approach says already_at_position, select its cross task next.
"""

PILOT = """You are the ONLY navigation controller of a stabilized indoor drone.
Choose one velocity command, brake, or camera turn. Commands expire after 0.6s.
The low-level servo handles gravity; it does NOT avoid obstacles or plan routes.
Depth is delayed, noisy and incomplete. clearance_m is observed safe travel in
each cardinal direction for the drone's volume. null means UNKNOWN, never clear.
Only translate toward a direction with measured clearance. First turn the camera
there if UNKNOWN; wait with brake while it turns or depth is unavailable. A look
command brakes and rotates the camera gradually. Single camera does not see behind.
For cruise require at least 1.3m clearance; for slow require at least 0.4m. If moving,
brake before switching axes or reversing. Brake on approach to the goal; never
continue along an axis after passing its goal coordinate. Slow is 0.25m/s; cruise
is the stated speed. Do not accelerate into a moving obstruction; brake or wait
when clearance is shrinking quickly. An uncertain or blinded drone should stop.
When blocked, you must choose a visible safe sideways or vertical detour, then
reassess the goal. Keep at least 0.4m from floor/ceiling. Do not indefinitely stare
in the final inspection direction while still far from the task's position.
If at_position is true, brake. Once stopped there, face the requested look direction
if one is given. Goal arrival is positional AND stopped. You cannot inspect or dock
from this controller; the mission planner handles those decisions after arrival.
motion_axis_needs_travel=false while moving means BRAKE NOW: you have already
reached/passed that axis's goal coordinate even if another axis still needs travel.
For a needed axis with remaining distance under 0.6m, brake then use slow movement.
"""


def room_at(position):
    return SMALL.room_at(position)


def cardinal(vector):
    axis = int(np.argmax(np.abs(vector)))
    return (("east", "west"), ("north", "south"), ("up", "down"))[axis][int(vector[axis] < 0)]


class Experiment:
    layout = SMALL
    def __init__(self, config, gateway, frame_dir=None, *, world=None, sensors=None):
        self.config = config
        self.gateway = gateway
        self.world = world if world is not None else World(config)
        self.layout = getattr(self.world, "layout", SMALL)
        self.sensors = sensors if sensors is not None else Sensors(self.world)
        self.stage = 0
        self.calls = []
        self.tasks = []
        self.reports = []
        self.memory = {}
        self.visits = Counter()
        self.task = None
        self.detour = None
        self.task_started = 0.
        self.best_distance = 100.
        self.last_progress = 0.
        self.last_observation = None
        self.violations = []
        self.stale = 0
        self.error_streak = 0
        self.error_started = None
        self.cost = 0.
        self.frame_dir = frame_dir
        if frame_dir:
            frame_dir.mkdir()
        self.last_saved_frame = -10.
        self.started_wall = time.perf_counter()
        self.sensors.advance(config.sensor_delay+.01)

    def observe(self):
        observation = self.sensors.observation()
        now = self.world.time
        for detection in observation["detections"]:
            self.memory[detection["marker"]] = detection
        pose = observation["position_estimate"]
        observation["room"] = self.layout.room_at(pose)
        observation["speed"] = round(float(np.linalg.norm(observation["velocity_estimate"])), 3)
        closing = {}
        if self.last_observation:
            elapsed = now-self.last_observation[0]
            for direction, clearance in observation["clearance_m"].items():
                previous = self.last_observation[1]["clearance_m"][direction]
                if clearance is not None and previous is not None and elapsed > .05:
                    closing[direction] = round(max(0., (previous-clearance)/elapsed), 2)
        observation["closing_rate_mps"] = closing
        self.last_observation = (now, observation)
        if self.frame_dir and now-self.last_saved_frame >= 2 and self.sensors.current:
            self.sensors.save_frame(self.frame_dir/f"{now:08.3f}.npz")
            self.last_saved_frame = now
        return observation

    def facts(self, observation):
        objective = self.world.mission[self.stage]
        pose = np.array(observation["position_estimate"])
        visible = [d for d in observation["detections"] if d["marker"] == objective.get("marker")]
        target = max(visible, key=lambda d: d["pixels"]) if visible else None
        distance = float(np.linalg.norm(pose-target["position"])) if target else None
        return {"objective": objective, "inspection_ready": bool(target and distance <= 1.5 and
                    observation["speed"] <= .15 and observation["room"] == objective.get("room")),
                "target_visible": target is not None, "target_distance": round(distance, 2) if distance is not None else None,
                "dock_ready": bool("dock" in objective and np.linalg.norm(pose-objective["dock"]) <= .45 and observation["speed"] <= .15)}

    def options(self, observation):
        pose = np.array(observation["position_estimate"])
        room = observation["room"]
        options = {"look_"+name: {"look": name} for name in DIRECTIONS}
        for name, center in self.layout.doors.items():
            first, second = self.layout.connections[name]
            if room not in (first, second):
                continue
            axis = self.layout.door_axis(name)
            destination = second if room == first else first
            direction = self.layout.door_direction(name, destination)
            near, far = np.array(center), np.array(center)
            near[axis] -= direction*1.0
            far[axis] += direction*1.0
            options["approach_"+name] = {"position": near.tolist(), "purpose": "approach doorway from current room"}
            if np.linalg.norm(pose-near) < 1.4:
                options["cross_"+name] = {"position": far.tolist(), "purpose": "cross doorway into "+(second if room == first else first)}
        objective = self.world.mission[self.stage]
        if room in self.layout.rooms and "marker" in objective:
            for name, view in self.layout.search_viewpoints().items():
                if view['room'] == room:
                    options[name] = {'position':list(view['position']), 'purpose':'search from a different viewpoint'}
        facts = self.facts(observation)
        objective = facts["objective"]
        observed = self.memory if self.config.memory else {d["marker"]: d for d in observation["detections"]}
        if objective.get("marker") in observed:
            target = np.array(observed[objective["marker"]]["position"])
            vector = target-pose
            direction = vector/max(.001, np.linalg.norm(vector))
            goal = target-direction*.95
            options["approach_marker"] = {"position": goal.tolist(), "look": cardinal(vector),
                                           "purpose": "approach last observed target; verify visually"}
        if "dock" in objective and room == self.layout.room_at(objective["dock"]):
            options["return_launch"] = {"position": objective["dock"], "purpose": "return to launch position"}
        if "marker" in objective:
            options["inspect"] = {"purpose": "report current requested marker inspected; requires inspection_ready=true"}
        else:
            options["dock"] = {"purpose": "finish mission at launch; requires dock_ready=true"}
        options["give_up"] = {"purpose": "honestly terminate incomplete"}
        for option in options.values():
            if "position" in option:
                distance = float(np.linalg.norm(pose-option["position"]))
                option.update(distance_m=round(distance, 2), already_at_position=distance <= .42)
        return options

    def request(self, role, state, instructions, criteria):
        request_time = self.world.time
        result = self.gateway.choose(state, instructions, criteria)
        self.cost += float(result.get("usage", {}).get("cost", 0) or 0)
        # A live simulator has already advanced while the network call was pending.
        if not getattr(self.world, "realtime", False):
            self.sensors.advance(result["latency_seconds"])
        log = {"role": role, "time": request_time, "response_time": self.world.time, **result}
        self.calls.append(log)
        if result.get("error"):
            self.error_streak += 1
            if self.error_started is None:
                self.error_started = time.monotonic()
            self.world.command("brake")
            failed = (time.monotonic()-self.error_started > 15 if getattr(self.config, "front_controls", False)
                      else self.error_streak >= 3)
            if failed and self.world.status == "running":
                self.world.status = "api_error"
            return None, log
        self.error_streak = 0
        self.error_started = None
        return result["choice"], log

    def plan(self, observation):
        self.world.command("brake")
        options = self.options(observation)
        facts = self.facts(observation)
        state = {"mission": self.world.mission, "stage": self.stage, **facts,
                 "ACTIVE_OBJECTIVE": facts["objective"] if "marker" in facts["objective"] else {"return_to_launch": facts["objective"]["dock"], "room": self.layout.room_at(facts["objective"]["dock"])},
                 "time_remaining": round(self.config.seconds-self.world.time, 1),
                 "sensors": observation, "known_room_bounds": self.layout.rooms, "known_doors": self.layout.doors,
                 "known_room_connections": self.layout.neighbors(),
                 "options": options}
        if self.config.memory:
            state.update(observed_markers=self.memory, recent_tasks=self.tasks[-12:], visits=dict(self.visits))
            if self.config.policy == "v4":
                failures = Counter(t["name"] for t in self.tasks if t["outcome"] in ("stalled", "replan_requested"))
                state["failed_navigation_attempts"] = dict(failures)
                state["routing_note"] = "A stalled cross_DOOR task means this doorway may be obstructed. Choose another connected room/door to bypass it. Repeating the same failed crossing is not progress."
        choice, log = self.request("planner", state, PLANNER,
                                    {name: json.dumps(option) for name, option in options.items()})
        if choice is None or self.world.status != "running":
            return
        if self.world.time >= self.config.seconds:
            return
        if choice in ("inspect", "dock"):
            # Both observed evidence and independent physical truth must support a report.
            fresh = self.observe()
            valid = self.facts(fresh)["inspection_ready" if choice == "inspect" else "dock_ready"]
            objective = self.world.mission[self.stage]
            if choice == "inspect":
                target = self.world.targets.get(objective.get("marker"))
                valid = valid and target is not None and np.linalg.norm(self.world.position-target["position"]) <= 1.75
            else:
                valid = valid and "dock" in objective and np.linalg.norm(self.world.position-self.world.start) <= .55
            valid = bool(valid and np.linalg.norm(self.world.velocity) <= .15)
            self.reports.append({"choice": choice, "stage": self.stage, "time": self.world.time,
                                 "valid": valid, "sensors": fresh, "position": self.world.position.tolist(),
                                 "target_ground_truth": self.world.targets.get(objective.get("marker"))})
            if not valid:
                self.world.status = "false_report"
            else:
                self.stage += 1
                if self.stage == len(self.world.mission):
                    self.world.status = "success"
            return
        if choice == "give_up":
            self.world.status = "incomplete"
            return
        self.task = {"name": choice, "mission_stage": self.stage, **options[choice]}
        self.detour = None
        self.task_started = self.world.time
        self.last_progress = self.world.time
        self.best_distance = self.distance(fresh=observation)
        self.tasks.append({**self.task, "time": self.world.time, "start": observation["position_estimate"], "outcome": "active"})

    def distance(self, fresh):
        if self.task and "position" in self.task:
            return float(np.linalg.norm(np.array(fresh["position_estimate"])-self.task["position"]))
        return 0.

    def task_finished(self, observation):
        if self.task is None:
            return True
        distance = self.distance(observation)
        if distance < self.best_distance-.08:
            self.best_distance = distance
            self.last_progress = self.world.time
        at_position = distance <= .42
        aim = self.task.get("look")
        aimed = not aim or np.dot(self.world.camera_direction, DIRECTIONS[aim]) > .99 or self.config.rig == "surround"
        outcome = None
        if at_position and observation["speed"] <= .15 and aimed:
            outcome = "arrived"
        elif self.world.time-self.task_started > 22 or self.world.time-self.last_progress > 8:
            outcome = "stalled"
        if outcome:
            self.tasks[-1].update(outcome=outcome, ended=self.world.time, end_position=observation["position_estimate"])
            self.visits[self.task["name"]] += 1
            self.task = None
            return True
        return False

    def control(self, observation):
        tactical = self.config.policy in ("v3", "v4")
        if self.detour is not None:
            reached = np.linalg.norm(np.array(observation["position_estimate"])-self.detour["position"]) <= .20
            if (reached and observation["speed"] <= .15) or self.world.time-self.detour["time"] > 10:
                self.detour = None
        mission_goal = self.task.get("position", observation["position_estimate"])
        goal = self.detour["position"] if self.detour else mission_goal
        delta = np.array(goal)-observation["position_estimate"]
        at_position = np.linalg.norm(delta) <= (.20 if self.detour else .42)
        state = {"sensors": observation, "goal": goal, "goal_offset": np.round(delta, 2).tolist(),
                 "at_position": bool(at_position), "look_when_at_position": self.task.get("look") if at_position else None,
                 "cruise_speed_mps": self.config.speed, "task": self.task["name"],
                 "recent_actions": [{"choice": c.get("choice"), "position": c["state"]["sensors"]["position_estimate"]}
                                    for c in self.calls[-6:] if c["role"] == "control"]}
        velocity = np.array(observation["velocity_estimate"])
        motion_direction = cardinal(velocity) if observation["speed"] > .15 else None
        along_motion = float(np.dot(delta, DIRECTIONS[motion_direction])) if motion_direction else None
        state.update(stopped=observation["speed"] <= .15, motion_direction=motion_direction,
                     remaining_along_motion=round(along_motion, 2) if along_motion is not None else None,
                     motion_axis_needs_travel=bool(along_motion is not None and along_motion > (.10 if self.detour else .20)),
                     axis_needs={name: bool(np.dot(delta, direction) > .20) for name, direction in DIRECTIONS.items()})
        criteria = {"brake": "Stop, wait for sensing/camera, or hold the reached goal."}
        for name in DIRECTIONS:
            criteria["look_"+name] = "Brake and gradually aim camera "+name+" to acquire depth before motion."
            criteria["slow_"+name] = "Move "+name+" at 0.25m/s; require measured clearance >=0.4m and no axis change while moving."
            criteria["cruise_"+name] = "Move "+name+" at cruise speed; require measured clearance >=1.3m and no axis change while moving."
        instructions = PILOT
        if tactical:
            state.update(mission_goal=mission_goal, active_detour=self.detour)
            state["depth_status"] = {
                name: "UNKNOWN" if value is None else "OPEN" if value >= 1.3 else "SLOW_ONLY" if value >= .4 else "BLOCKED"
                for name, value in observation["clearance_m"].items()}
            instructions += "\nDepth status is decisive: UNKNOWN/BLOCKED prohibits translation there; SLOW_ONLY prohibits cruise. Brake immediately when current movement is BLOCKED.\n"
            instructions += """\nIf the goalward direction is blocked, choose a detour task instead of repeatedly
braking without progress. detour_DIRECTION brakes and sets a temporary goal 1.3m
that way; it does not automatically fly. Then YOU choose each camera/movement
command toward that temporary goal using the same depth rules. It can intentionally
move away from the mission goal to go around or above an obstacle. After reaching
and stopping at the temporary goal, reconsider the original mission goal. Some
obstacles need more than one detour step. Example: east blocked by a beam, north
appears open -> detour_north; look_north if needed; move using measured north depth;
stop at the temporary goal; look_east and reassess. Never translate into UNKNOWN.
"""
            if self.detour is None:
                for name in DIRECTIONS:
                    criteria["detour_"+name] = "Brake and choose a temporary 1.3m goal "+name+" to go around an obstruction; movement is decided on later calls."
                    if self.config.policy == "v4":
                        status = state["depth_status"][name]
                        criteria["detour_"+name] += (
                            f" Observed direction is {status}. This is a viable bypass candidate." if status == "OPEN" else
                            f" Observed direction is {status}: DO NOT SELECT this detour. It cannot bypass a known obstruction in that same direction. Look elsewhere for OPEN space first.")
                if self.config.policy == "v4":
                    instructions += "\nA detour must use an OPEN direction. Never select detour_east when east is BLOCKED: that just sets another goal behind the same obstacle. Choose an OPEN sideways/up/down bypass; look first if no bypass direction is observed. Failed navigation attempts require a changed route, not repeated identical intent.\n"
        if self.config.tracking:
            instructions += "\nVisual motion tracks predict nearby moving obstacles. If current_course_risk=true while moving, BRAKE NOW. When stopped, choose a safe detour/escape direction whose predicted risk is false; consider flying above a moving obstacle when up is observed OPEN. Do not choose movement with predicted_risk_by_direction=true even if current static depth is OPEN. These are uncertain measurements, not ground-truth actor trajectories.\n"
            if self.config.policy == "v4":
                for name, risk in observation["predicted_risk_by_direction"].items():
                    if risk:
                        for prefix in ("cruise_", "slow_", "detour_"):
                            if prefix+name in criteria:
                                criteria[prefix+name] += " PREDICTED MOVING-OBSTACLE RISK: do not select; choose brake or an observed safe alternative."
        choice, log = self.request("control", state, instructions, criteria)
        accepted = choice is not None and self.world.status == "running" and self.world.time < self.config.seconds and log["latency_seconds"] <= self.world.lease
        if accepted:
            accepted = self.apply_control(choice, observation, log)
        else:
            self.world.command("brake")
            if choice is not None and log["latency_seconds"] > self.world.lease:
                self.stale += 1
        log["accepted"] = accepted
        self.sensors.advance(max(0., .2-log["latency_seconds"]))

    def apply_control(self, choice, observation, log):
        self.audit(choice, observation, log)
        if choice.startswith("detour_"):
            self.select_detour(choice, observation)
            self.world.command("brake")
        else:
            self.world.command(choice)
        return True

    def select_detour(self, choice, observation):
        direction = DIRECTIONS[choice[7:]]
        self.detour = {"position": (np.array(observation["position_estimate"])+1.3*np.array(direction)).tolist(), "time": self.world.time,
                       "selected_by": "jev", "choice": choice}
        self.last_progress = self.world.time

    def audit(self, choice, observation, log):
        if not choice.startswith(("slow_", "cruise_")):
            return
        speed, direction = choice.split("_", 1)
        clearance = observation["clearance_m"][direction]
        reasons = []
        if clearance is None or clearance < (.4 if speed == "slow" else 1.3):
            reasons.append("unknown_or_insufficient_observed_clearance")
        velocity = np.array(observation["velocity_estimate"])
        if np.linalg.norm(velocity) > .15 and np.dot(velocity/np.linalg.norm(velocity), DIRECTIONS[direction]) < .95:
            reasons.append("changed_axis_while_moving")
        if reasons:
            self.violations.append({"time": log["time"], "choice": choice, "reasons": reasons})

    def run(self):
        try:
            while self.world.status == "running" and self.world.time < self.config.seconds and self.cost < self.config.budget:
                if not getattr(self.world, "telemetry_ready", True):
                    self.sensors.advance(.1)
                    continue
                observation = self.observe()
                if self.task_finished(observation):
                    self.plan(observation)
                else:
                    self.control(observation)
            if self.world.status == "running":
                self.world.status = "budget" if self.cost >= self.config.budget else "timeout"
            return {"world": self.world.metadata(), "status": self.world.status, "stage": self.stage,
                    "sim_seconds": self.world.time, "wall_seconds": time.perf_counter()-self.started_wall,
                    "cost_usd": self.cost, "calls": self.calls, "tasks": self.tasks, "reports": self.reports,
                    "violations": self.violations, "stale_responses": self.stale,
                    "telemetry_outages": getattr(self.world, "telemetry_outages", []),
                    "trace": self.world.trace, "contacts": self.world.contacts,
                    "min_clearance": self.world.min_clearance, "near_miss_seconds": self.world.near_miss_seconds,
                    "blackout_frames": self.sensors.blackout_frames, "observed_markers": self.memory}
        finally:
            self.world.close()
