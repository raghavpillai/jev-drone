"""Jev chooses body-relative velocity/yaw patches from front-camera observations."""

import json
import math

import numpy as np

from jev_drone.control.axis_progress import INSTRUCTIONS as AXIS_PROGRESS_INSTRUCTIONS
from jev_drone.control.axis_progress import describe as describe_axis_progress
from jev_drone.control.controls import BODY_CHANNELS, changed_channels
from jev_drone.control.decision_instructions import BRAKING, braking_facts
from jev_drone.control.doorway_control import INSTRUCTIONS as DOORWAY_INSTRUCTIONS
from jev_drone.control.doorway_control import describe as describe_doorway
from jev_drone.control.doorway_control import heading as doorway_heading
from jev_drone.control.doorway_phase import INSTRUCTIONS as DOOR_PHASE_INSTRUCTIONS
from jev_drone.control.doorway_phase import describe as describe_doorway_phase
from jev_drone.control.goal_phase import INSTRUCTIONS as GOAL_PHASE_INSTRUCTIONS
from jev_drone.control.goal_phase import describe as describe_goal_phase
from jev_drone.control.progress_facts import INSTRUCTIONS as PROGRESS_INSTRUCTIONS
from jev_drone.control.progress_facts import enrich
from jev_drone.control.stopping import required_clearance
from jev_drone.perception.depth import clearance as fused_clearance
from jev_drone.planning.agent import Experiment
from jev_drone.planning.detour_route import INSTRUCTIONS as DETOUR_INSTRUCTIONS
from jev_drone.planning.detour_route import advance as advance_detour
from jev_drone.planning.detour_route import candidate as detour_candidate
from jev_drone.planning.detour_route import observed_obstacles
from jev_drone.planning.door_memory import INSTRUCTIONS as DOOR_INSTRUCTIONS
from jev_drone.planning.door_memory import DoorMemory
from jev_drone.planning.planner_modes import INSTRUCTIONS as MODE_INSTRUCTIONS
from jev_drone.planning.planner_modes import available_modes, eligible
from jev_drone.planning.return_intent import describe as describe_return_intent
from jev_drone.planning.return_memory import INSTRUCTIONS as RETURN_INSTRUCTIONS
from jev_drone.planning.return_memory import ReturnMemory
from jev_drone.planning.room_intent import INSTRUCTIONS as ROOM_INSTRUCTIONS
from jev_drone.planning.room_intent import describe as describe_room_intent
from jev_drone.planning.room_transition import INSTRUCTIONS as TRANSITION_INSTRUCTIONS
from jev_drone.planning.room_transition import describe as describe_room_transition
from jev_drone.planning.search_memory import INSTRUCTIONS as SEARCH_INSTRUCTIONS
from jev_drone.planning.search_memory import SearchMemory
from jev_drone.world.world import DIRECTIONS

CHANNELS = BODY_CHANNELS


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def patches(speed):
    zero = dict.fromkeys(CHANNELS, 0.0)
    actions = {
        "keep_controls": {},
        "brake": zero,
        "brake_and_replan": zero,
        "forward_creep": {"forward_mps": 0.1},
        "backward_creep": {"forward_mps": -0.1},
        "strafe_left_creep": {"right_mps": -0.1},
        "strafe_right_creep": {"right_mps": 0.1},
        "up_creep": {"up_mps": 0.1},
        "down_creep": {"up_mps": -0.1},
        "forward_slow": {"forward_mps": 0.25},
        "forward_full": {"forward_mps": speed},
        "backward_slow": {"forward_mps": -0.25},
        "backward_full": {"forward_mps": -speed},
        "stop_forward": {"forward_mps": 0.0},
        "strafe_left_slow": {"right_mps": -0.25},
        "strafe_right_slow": {"right_mps": 0.25},
        "strafe_left_full": {"right_mps": -speed},
        "strafe_right_full": {"right_mps": speed},
        "stop_strafe": {"right_mps": 0.0},
        "pan_left": {"yaw_rate_rps": 0.6},
        "pan_right": {"yaw_rate_rps": -0.6},
        "pan_left_slow": {"yaw_rate_rps": 0.2},
        "pan_right_slow": {"yaw_rate_rps": -0.2},
        "stop_pan": {"yaw_rate_rps": 0.0},
        "up": {"up_mps": 0.25},
        "down": {"up_mps": -0.25},
        "stop_vertical": {"up_mps": 0.0},
    }
    # Ordinary motion choices specify a complete stick state. Partial releases
    # and keep remain explicit; the executor encodes every choice as a delta.
    for name, assignment in actions.items():
        if name != "keep_controls" and not name.startswith("stop_"):
            actions[name] = {**zero, **assignment}
    for direction, forward in (("forward", 0.25), ("backward", -0.25)):
        for pan, rate in (("left", 0.2), ("right", -0.2)):
            actions[f"arc_{direction}_{pan}"] = {
                **zero,
                "forward_mps": forward,
                "yaw_rate_rps": rate,
            }
    for strafe, right in (("left", -0.25), ("right", 0.25)):
        for pan, rate in (("left", 0.2), ("right", -0.2)):
            actions[f"arc_strafe_{strafe}_pan_{pan}"] = {
                **zero,
                "right_mps": right,
                "yaw_rate_rps": rate,
            }
    return actions


def movement_reasons(values, observation, combined_clearance=None, limits=None):
    reasons = []
    limits = limits or {"slow_clearance_m": 0.4, "full_clearance_m": 1.3}
    gap = observation.get("nearest_observed_body_gap_m")
    if values["yaw_rate_rps"] and gap is not None and gap < 0.05:
        reasons.append("rotation_close_to_observed_obstacle")
    for channel, positive, negative in (
        ("forward_mps", "forward", "backward"),
        ("right_mps", "right", "left"),
        ("up_mps", "up", "down"),
    ):
        value = values.get(channel, 0.0)
        if not value:
            continue
        direction = positive if value > 0 else negative
        clearance = observation["body_clearance_m"][direction]
        measured = observation["body_velocity_mps"][positive]
        threshold = required_clearance(
            value, measured, observation.get("frame_age_seconds"), limits
        )
        if clearance is None or clearance < threshold:
            reasons.append("insufficient_or_unknown_" + direction + "_depth")
        if abs(measured) > 0.15 and measured * value < 0:
            reasons.append("must_brake_before_reversing_" + positive)
    speed = math.hypot(values["forward_mps"], values.get("right_mps", 0.0))
    if speed > 0.25 and (
        abs(values["yaw_rate_rps"]) > 0.1 or abs(observation["yaw_rate_rps"]) > 0.12
    ):
        reasons.append("cruise_while_turning")
    translation = [values.get(c, 0.0) for c in CHANNELS[:3]]
    combined_threshold = required_clearance(
        float(np.linalg.norm(translation)),
        float(np.linalg.norm(list(observation["body_velocity_mps"].values()))),
        observation.get("frame_age_seconds"),
        limits,
    )
    if sum(v != 0 for v in translation) > 1 and (
        combined_clearance is None or combined_clearance < combined_threshold
    ):
        reasons.append("insufficient_or_unknown_combined_depth")
    return reasons


PILOT = """You are Jev, the ONLY navigation controller of an indoor drone.
Select one explicit stick-state intent. Its transmitted patch contains only changed
channels; omitted channels KEEP their values. Ordinary move/pan intents release
the other axes. arc_forward_left/right and arc_backward_left/right explicitly
combine slow translation and pan; arc_strafe_DIRECTION_pan_DIRECTION combines
sideways translation and pan. stop_AXIS releases only that axis, keeping the
others. keep_controls retains all current axes. Read resulting_controls for the
complete state: this is what the drone will hold, regardless of patch length.
Positive forward_mps flies in the CURRENT heading; negative flies backward.
Positive right_mps strafes RIGHT, negative strafes LEFT, without turning the camera.
Positive yaw_rate_rps pans LEFT, negative pans RIGHT. This physically turns the
drone and its front camera, in radians/second. up_mps controls vertical velocity.
PX4 handles stabilization only. No algorithm chooses steering or avoids obstacles.
keep_controls is an explicit heartbeat; silence expires ALL controls after
control_limits.command_lease_seconds. Observe the declared control_limits.
brake zeros all four axes. stop_forward preserves strafe, vertical and pan. stop_pan preserves
translation. Read resulting_controls: do not accidentally keep a turn running!
creep choices move at 0.1 m/s, slow at 0.25 m/s. Use creep for tight clearances or
fine adjustments when slow would violate the stopping margin; prefer slow/full
when there is ample observed space. Creep still needs observed clearance.
brake_and_replan stops and explicitly returns this failed local task to the mission
planner. Use it if the local route remains blocked or an active bypass cannot
continue. Do not use it for ordinary braking, heading alignment or a brief missing
frame. It neither reports an objective nor advances the mission.
For a transition between axes, up/down releases horizontal controls and pan while
starting vertical motion. Do not keep an axis that already reached its goal. Prefer a combined
stop-old-axis/start-new-axis update when changing movement direction. Each choice
sends only values that change; repeating a satisfied choice renews the lease with
an empty patch. Its resulting_controls still describes the entire held state.

Only front-camera pixels reveal targets. Rear/left/right/up/down DEPTH supports clearance
but cannot find objects. Previously observed targets are memories,
not proof they are visible now. UNKNOWN depth is not free space.
Never select an option with movement_violations. Minimum clearance is given in
control_limits, with extra stopping allowance for speed, measurement age and
request/lease delay. Brake before reversing physical velocity.
When panning, at most slow translation is allowed; braking then turning in place
is easier near obstacles. Backward motion must use rear depth, not front depth.

goal_bearing_error_degrees is positive when the desired heading is to your LEFT.
body_goal_offset_m gives remaining forward, right and up distances in your current
heading. Negative forward is behind you; negative right is to your left. You may
move backward or strafe directly toward a goal without first turning. For short
sideways corrections strafe instead of repeatedly turning. If required_heading_degrees
is set, hold that heading while translating; yaw must not replace strafing.
For a requested turn, brake and pan in place. Under about 25 degrees use slow
pan to avoid overshooting; under 8 degrees stop pan. Stop panning before full
forward speed. Pan can be combined with slow forward motion when observed clear.
Move toward the selected goal. Read travel_effect: stop a horizontal axis when its
remaining distance is under 0.2m, a vertical axis under 0.1m, or when motion moves
away from the goal. Arrival must satisfy BOTH horizontal and altitude tolerances.
Use slow near the
goal, brake early within about 0.5m. If at_position,
stop translation; face requested_heading if present, then stop pan too. Arrival
requires low measured linear AND angular speed. Control setpoints are not measured
velocity: stopping a control takes time to stop the vehicle.

When translation is stopped and you are already panning toward a distant heading,
KEEP PANNING until close to alignment. Do NOT alternate pan and brake each call.
Turning in place does not require forward depth to be clear: the clearance rules
apply to TRANSLATION. A nonzero yaw control is not itself a reason to brake.
However, a rotor can hit nearby furniture or a wall WHILE TURNING IN PLACE.
nearest_observed_body_gap_m measures the gap from the 0.4m body envelope to the
closest observed depth point. If below 0.05m, do not turn: stop pan and translate
away using a direction with verified clearance, then turn once space is available.
Null does not certify safety; this proximity signal cannot reveal unseen objects.
Read turn_effect for each choice. Stopping pan while the heading is still far
away prevents progress; continuing a correctly directed turn is the useful action.

If forward is blocked, look around by panning. Choose an observed bypass or a
detour task; detour sets a temporary goal and brakes, it NEVER flies automatically.
Use up/down only with measured corresponding clearance. Do not keep pushing into
an obstacle, repeat the same failed attempt, or oscillate forever. When searching,
use recent_front_views to avoid repeatedly checking the same direction/position.
You cannot report inspection/docking here; the planner does that after arrival.
"""


class FrontExperiment(Experiment):
    mission_handoffs = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.front_views = []
        self.best_heading_error = math.inf
        self.detour_history = []
        self.target_notified_stage = None
        self.search_memory = SearchMemory(self.layout)
        self.return_memory = ReturnMemory()
        self.door_memory = DoorMemory(self.layout)

    def observe(self):
        observation = super().observe()
        if self.mission_handoffs:
            self.search_memory.observe(observation, self.world.time)
            self.return_memory.observe(observation, "dock" in self.world.mission[self.stage])
            self.door_memory.observe(self.sensors.observed_cloud, self.world.time)
        if observation["front_frame_available"] and observation["valid_depth_fraction"] > 0.4:
            view = {
                "time": round(self.world.time, 2),
                "position": observation["position_estimate"],
                "yaw_degrees": observation["yaw_degrees"],
                "room": observation["room"],
                "markers": sorted({d["marker"] for d in observation["detections"]}),
            }
            last = self.front_views[-1] if self.front_views else None
            if (
                last is None
                or np.linalg.norm(np.array(view["position"]) - last["position"]) > 0.7
                or abs(wrap(math.radians(view["yaw_degrees"] - last["yaw_degrees"])))
                > math.radians(20)
            ):
                self.front_views.append(view)
                self.front_views = self.front_views[-48:]
        return observation

    def request(self, role, state, instructions, criteria):
        if role == "control":
            state, criteria = enrich(state, criteria)
            state, criteria = braking_facts(state, criteria)
            state, criteria = describe_doorway(state, criteria, self.layout)
            state, criteria = describe_doorway_phase(state, criteria)
            state, criteria = describe_goal_phase(state, criteria)
            state, criteria = describe_axis_progress(state, criteria)
            instructions += (
                PROGRESS_INSTRUCTIONS
                + BRAKING
                + DETOUR_INSTRUCTIONS
                + DOORWAY_INSTRUCTIONS
                + GOAL_PHASE_INSTRUCTIONS
            )
            if "near_goal_axis_progress" in state:
                instructions += AXIS_PROGRESS_INSTRUCTIONS
            instructions += "\nRead mission_progress before refining the waypoint. If target_in_inspection_range or dock_in_range is true, brake and stay stopped so the planner can report the observed objective. The waypoint is a means to reach the item, not an extra mission requirement. Do not move away from a currently visible close target to reach an arbitrary approach coordinate. Safety and braking rules still apply.\n"
            instructions += "\nIf goal_endpoint_observed_occupied is true, the selected waypoint intersects or is too close to an observed surface. Once stopped, choose brake_and_replan so a different endpoint can be selected; do not try to force arrival inside furniture. A positive endpoint gap does not certify unseen space.\n"
            if "doorway_phase" in state:
                instructions += DOOR_PHASE_INSTRUCTIONS
        if role == "planner":
            state = {**state, "recent_front_views": self.front_views[-24:]}
            state, criteria = self.door_memory.describe(state, criteria)
            instructions += DOOR_INSTRUCTIONS
            for name, option in state["options"].items():
                if option.get("observed_endpoint_unsafe") and name in criteria:
                    criteria[name] = (
                        "DO NOT SELECT this endpoint: observed depth places it in/too close to furniture or a wall. Choose a different position. "
                        + criteria[name]
                    )
            if "dock" in state["objective"]:
                state["return_route_memory"] = self.return_memory.describe()
                state, criteria = describe_return_intent(state, criteria)
                instructions += RETURN_INSTRUCTIONS
            state, criteria = describe_room_intent(state, criteria)
            instructions += ROOM_INSTRUCTIONS
            state, criteria = describe_room_transition(state, criteria, self.layout)
            if "last_completed_room_transition" in state:
                instructions += TRANSITION_INSTRUCTIONS
            if "marker" in state["objective"]:
                state, criteria = self.search_memory.describe(state, criteria)
                instructions += SEARCH_INSTRUCTIONS
            instructions += "\nTransit tasks are intermediate positions, not object searches. If return_launch or a doorway approach stalls, select an open intermediate transit point around the furniture, then resume the objective. Repeating the identical blocked direct route is not a new plan. An intermediate point may initially move away from the destination. observed_path_clearance_m describes ONLY the initial part of the straight path visible in current depth frames, not guaranteed clearance all the way. Prefer an observed open direction; null is unknown. Higher transit points may pass above low furniture; keep that altitude until past it instead of descending immediately onto the obstacle.\n"
            instructions += "\nSeeing a target does not imply a direct approach is reachable: it may be behind a partition. If approaches stall, use room viewpoint positions as intermediate transit steps around the partition, even if already visited. First reach the open end of the obstacle, then move past it while retaining that lateral offset or altitude, and ONLY THEN approach the item. Do not abandon a useful bypass halfway to cut diagonally back into the same obstruction. Choosing another target-side standoff repeatedly from the blocked side is not a route.\n"
            instructions += "\nInspection and docking are outcomes, not exact waypoints. If inspection_ready is true, select inspect NOW; do not keep refining an approach. If dock_ready is true, select dock NOW. A target_observed task outcome interrupts searching so you can approach the newly seen target. When a cross_DOOR option is available toward the required room, you may select it without first reaching the approach coordinate exactly. Prefer the nearest observed-clear target approach on your current side; do not fly around a visible reachable item just to reach a far-side waypoint.\n"
            instructions += "\nOnly a 137.5-degree FRONT camera detects objects. Look tasks physically yaw the drone. Objects outside its field of view are unknown until seen; auxiliary depth never supplies object detections. Observe different headings as well as positions when searching. The pilot uses forward/backward, strafe left/right, up/down and pan left/right. Target approaches are horizontal viewpoints at the observed item height because the camera cannot pitch downward. If an approach stalls or its observed_goal_gap_m is negative, try another side of the item. Positive gap is only observed evidence, not proof of unobserved free space.\n"
            mode, mode_log = super().request(
                "planner_mode", state, MODE_INSTRUCTIONS, available_modes(state)
            )
            if mode is None:
                return None, mode_log
            criteria = {name: text for name, text in criteria.items() if eligible(name, mode)}
            state = {**state, "selected_task_kind": mode}
            instructions += (
                "\nYou selected task kind "
                + mode
                + ". Select the concrete task now, using current evidence. All controls will be chosen separately by Jev."
            )
        return super().request(role, state, instructions, criteria)

    def options(self, observation):
        options = super().options(observation)
        objective = self.world.mission[self.stage]
        room = observation["room"]
        if room in self.layout.rooms and "marker" in objective:
            lo, hi = map(np.array, self.layout.rooms[room])
            center = ((lo + hi) / 2).tolist()
            if np.linalg.norm(np.array(center[:2]) - observation["position_estimate"][:2]) > 0.25:
                options["look_into_room"] = {
                    "look_position": center,
                    "purpose": "Look into the current room from here, without selecting a translation.",
                }
            for name, option in options.items():
                if name.startswith("view_"):
                    option["look_position"] = [center[0], center[1], option["position"][2]]
                    option["purpose"] = (
                        "Move to this viewpoint THEN look into the room. Passing near its position alone is not a survey."
                    )
        if observation["room"] != objective.get("room", observation["room"]):
            options.pop("approach_marker", None)
        for name, option in options.items():
            crossing_heading = doorway_heading({"name": name, **option}, self.layout)
            if crossing_heading is not None:
                option["heading_degrees"] = crossing_heading
            if "position" in option:
                option["already_at_position"] = self.at_goal(
                    np.array(option["position"]) - observation["position_estimate"]
                )
            if name.startswith(("approach_", "cross_")) and name != "approach_marker":
                rooms = self.layout.connections[name.split("_", 1)[1]]
                option["leads_to_room"] = next(r for r in rooms if r != observation["room"])
        for name in ("look_up", "look_down"):
            options.pop(name, None)  # Body yaw cannot pitch a camera vertically.
        if "dock" in objective:
            options.update(self.return_memory.options(observation, self.tasks))
        if "dock" in objective or observation["room"] != objective.get("room"):
            for name, view in self.search_memory.viewpoints.items():
                if view["room"] == observation["room"]:
                    options[name.replace("view_", "transit_", 1)] = {
                        "position": list(view["position"]),
                        "purpose": "Intermediate transit position around furniture toward the next room or launch; no search or inspection.",
                    }
        if "approach_marker" in options:
            objective = self.world.mission[self.stage]
            detection = self.memory.get(objective.get("marker"))
            if detection:
                options.pop("approach_marker")
                target = np.array(detection["position"])
                pose = np.array(observation["position_estimate"])
                toward_drone = pose - target
                toward_drone[2] = 0.0
                directions = {
                    name: np.array(DIRECTIONS[name], dtype=float)
                    for name in ("east", "west", "north", "south")
                }
                if np.linalg.norm(toward_drone) > 0.1:
                    directions["current_side"] = toward_drone / np.linalg.norm(toward_drone)
                lo, hi = map(np.array, self.layout.rooms[objective["room"]])
                for side, direction in directions.items():
                    goal = target + 0.95 * direction
                    if np.any(goal < lo + 0.45) or np.any(goal > hi - 0.45):
                        continue
                    name = "approach_marker_" + side
                    options[name] = {
                        "position": goal.tolist(),
                        "look_position": target.tolist(),
                        "purpose": "Inspect from the "
                        + side
                        + " at observed item height; front camera cannot look straight down.",
                        "observed_goal_gap_m": self.sensors.observed_gap_at(goal),
                        "distance_m": round(float(np.linalg.norm(goal - pose)), 2),
                        "already_at_position": self.at_goal(goal - pose),
                    }
        pose = np.array(observation["position_estimate"])
        for option in options.values():
            if "position" in option:
                delta = np.array(option["position"]) - pose
                option["distance_m"] = round(float(np.linalg.norm(delta)), 2)
                option["already_at_position"] = self.at_goal(delta)
                gap = self.sensors.observed_gap_at(np.array(option["position"]))
                option["observed_goal_gap_m"] = gap
                option["observed_endpoint_unsafe"] = gap is not None and gap < 0.08
                option["observed_path_clearance_m"] = (
                    fused_clearance(self.sensors.current, pose, delta.copy())
                    if np.linalg.norm(delta) > 0.1
                    else None
                )
        # An already completed stationary turn would finish immediately without
        # acquiring a new view. Keep meaningful turns and every positional task.
        fresh_front = (
            observation.get("front_frame_available")
            and observation.get("front_valid_depth_fraction", 0.0) > 0.4
            and observation.get("frame_age_seconds") is not None
            and observation["frame_age_seconds"] <= 0.65
        )
        settled = observation["speed"] <= 0.10 and abs(observation["yaw_rate_rps"]) <= 0.10
        if fresh_front and settled:
            for name, option in list(options.items()):
                if not name.startswith("look_") or "position" in option:
                    continue
                heading = self.requested_heading(observation, option)
                if heading is not None and abs(
                    wrap(heading - math.radians(observation["yaw_degrees"]))
                ) <= math.radians(12):
                    del options[name]
        return options

    def facts(self, observation):
        facts = super().facts(observation)
        settled = observation["speed"] <= 0.10 and abs(observation["yaw_rate_rps"]) <= 0.10
        facts["inspection_ready"] &= settled
        facts["dock_ready"] &= settled
        return facts

    def plan(self, observation):
        super().plan(observation)
        self.best_heading_error = math.inf
        self.progress_heading_phase = None

    def requested_heading(self, observation, task=None):
        task = self.task if task is None else task
        if task.get("heading_degrees") is not None:
            return math.radians(task["heading_degrees"])
        if task.get("look_position") is not None:
            delta = np.array(task["look_position"]) - observation["position_estimate"]
            return math.atan2(delta[1], delta[0])
        if task.get("look"):
            direction = DIRECTIONS[task["look"]]
            return math.atan2(direction[1], direction[0])
        return None

    def at_goal(self, delta, detour=False):
        horizontal = 0.25 if detour else self.config.goal_horizontal_tolerance_m
        return bool(
            np.linalg.norm(delta[:2]) <= horizontal
            and abs(delta[2]) <= self.config.goal_vertical_tolerance_m
        )

    def desired_heading(self, observation, goal, at_position):
        delta = np.array(goal) - observation["position_estimate"]
        if self.task.get("heading_degrees") is not None:
            return self.requested_heading(observation), "fixed_heading"
        if at_position:
            return self.requested_heading(observation), "final_heading"
        tolerance = 0.25 if self.detour else self.config.goal_horizontal_tolerance_m
        if np.linalg.norm(delta[:2]) <= tolerance:
            return None, "altitude"
        return math.atan2(delta[1], delta[0]), "travel"

    def crossed_doorway(self, observation):
        name = self.task["name"]
        if not name.startswith("cross_") or name[6:] not in self.layout.doors:
            return False
        door = name[6:]
        axis = self.layout.door_axis(door)
        center = self.layout.doors[door][axis]
        direction = 1 if self.task["position"][axis] > center else -1
        destination = next(
            room
            for room in self.layout.connections[door]
            if self.layout.door_direction(door, room) == direction
        )
        return (
            observation["room"] == destination
            and direction * (observation["position_estimate"][axis] - center) >= 0.65
        )

    def task_finished(self, observation):
        if self.task is None:
            return True
        event = self.mission_event(observation)
        if event:
            self.tasks[-1].update(
                outcome=event, ended=self.world.time, end_position=observation["position_estimate"]
            )
            self.task = self.detour = None
            return True
        distance = self.distance(observation)
        delta = (
            np.array(self.task.get("position", observation["position_estimate"]))
            - observation["position_estimate"]
        )
        at_position = self.at_goal(delta) or self.crossed_doorway(observation)
        aim = self.requested_heading(observation)
        error = (
            abs(wrap(aim - math.radians(observation["yaw_degrees"]))) if aim is not None else 0.0
        )
        goal = (
            self.detour["position"]
            if self.detour
            else self.task.get("position", observation["position_estimate"])
        )
        goal_reached = self.at_goal(
            np.array(goal) - observation["position_estimate"], detour=self.detour is not None
        ) or self.crossed_doorway(observation)
        desired, aligning = self.desired_heading(observation, goal, goal_reached)
        phase = (tuple(self.detour["position"]) if self.detour else "waypoint", aligning)
        if phase != getattr(self, "progress_heading_phase", None):
            self.best_heading_error = math.inf
            self.progress_heading_phase = phase
        travel_error = (
            abs(wrap(desired - math.radians(observation["yaw_degrees"])))
            if desired is not None
            else 0.0
        )
        if distance < self.best_distance - 0.08 or travel_error < self.best_heading_error - 0.15:
            self.last_progress = self.world.time
            self.best_distance = min(self.best_distance, distance)
            self.best_heading_error = min(self.best_heading_error, travel_error)
        if self.detour is not None:
            delta = np.array(self.detour["position"]) - observation["position_estimate"]
            detour_distance = float(np.linalg.norm(delta))
            angle = abs(
                wrap(math.atan2(delta[1], delta[0]) - math.radians(observation["yaw_degrees"]))
            )
            if (
                detour_distance < self.detour.get("best_distance", math.inf) - 0.08
                or angle < self.detour.get("best_heading_error", math.inf) - 0.15
            ):
                self.last_progress = self.world.time
                self.detour["best_distance"] = min(
                    detour_distance, self.detour.get("best_distance", math.inf)
                )
                self.detour["best_heading_error"] = min(
                    angle, self.detour.get("best_heading_error", math.inf)
                )
        outcome = None
        if (
            at_position
            and error <= math.radians(12)
            and observation["speed"] <= 0.10
            and abs(observation["yaw_rate_rps"]) <= 0.10
        ):
            outcome = "arrived"
        elif self.world.time - self.task_started > 90 or self.world.time - self.last_progress > 12:
            outcome = "stalled"
        if outcome:
            self.tasks[-1].update(
                outcome=outcome,
                ended=self.world.time,
                end_position=observation["position_estimate"],
            )
            self.visits[self.task["name"]] += 1
            self.task = None
        return outcome is not None

    def mission_event(self, observation):
        if not self.mission_handoffs:
            return None
        facts = self.facts(observation)
        if facts["inspection_ready"] or facts["dock_ready"]:
            return "objective_ready"
        if (
            facts["target_visible"]
            and self.task["name"].startswith(("view_", "look_"))
            and self.target_notified_stage != self.stage
        ):
            self.target_notified_stage = self.stage
            return "target_observed"
        return None

    def finish_for_replan(self, observation):
        self.tasks[-1].update(
            outcome="replan_requested",
            ended=self.world.time,
            end_position=observation["position_estimate"],
        )
        self.visits[self.task["name"]] += 1
        self.task = self.detour = None

    def control(self, observation):
        pose = np.array(observation["position_estimate"])
        heading = math.radians(observation["yaw_degrees"])
        if self.detour is not None:
            reached = (
                self.at_goal(pose - self.detour["position"], detour=True)
                and observation["speed"] <= 0.1
            )
            if reached or self.world.time - self.detour["time"] > 45:
                self.detour_history.append(
                    {
                        **self.detour,
                        "outcome": "arrived" if reached else "expired",
                        "end_position": pose.tolist(),
                        "ended": self.world.time,
                    }
                )
                self.detour = advance_detour(self.detour, self.world.time) if reached else None
                if self.detour is not None:
                    self.last_progress = self.world.time
        goal = self.detour["position"] if self.detour else self.task.get("position", pose.tolist())
        delta = np.array(goal) - pose
        passage_complete = self.crossed_doorway(observation)
        at_position = self.at_goal(delta, detour=self.detour is not None) or passage_complete
        desired, _ = self.desired_heading(observation, goal, at_position)
        error = wrap(desired - heading) if desired is not None else 0.0
        snapshot = self.world.flight.control_snapshot()
        current = snapshot["body_controls"]
        choices = patches(self.config.speed)
        detours = {}
        if self.detour is None:
            fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
            left = np.array([-math.sin(heading), math.cos(heading), 0.0])
            for name, direction in {
                "left": left,
                "right": -left,
                "forward": fwd,
                "backward": -fwd,
                "up": np.array([0.0, 0.0, 1.0]),
                "down": np.array([0.0, 0.0, -1.0]),
            }.items():
                for prefix, distance in (
                    ("detour_", 1.3),
                    ("detour_medium_", 0.9),
                    ("detour_short_", 0.6),
                ):
                    choices[prefix + name] = dict.fromkeys(CHANNELS, 0.0)
                    detours[prefix + name] = detour_candidate(
                        pose,
                        self.task.get("position", pose),
                        direction,
                        distance,
                        name in ("left", "right", "up", "down"),
                        self.layout,
                    )
        choices = {name: changed_channels(current, patch) for name, patch in choices.items()}
        effects = {
            name: {"patch": patch, "resulting_controls": {**current, **patch}}
            for name, patch in choices.items()
        }
        fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
        right = np.array([math.sin(heading), -math.cos(heading), 0.0])
        body_delta = dict(
            forward=round(float(delta @ fwd), 2),
            right=round(float(delta @ right), 2),
            up=round(float(delta[2]), 2),
        )
        combined = {}
        limits = {
            name: getattr(self.config, name)
            for name in (
                "command_lease_seconds",
                "creep_clearance_m",
                "slow_clearance_m",
                "full_clearance_m",
                "stopping_margin",
                "braking_settling_allowance_s",
            )
        }
        for name, effect in effects.items():
            values = effect["resulting_controls"]
            translation = [values[c] for c in CHANNELS[:3]]
            if sum(v != 0 for v in translation) > 1:
                key = tuple(translation)
                if key not in combined:
                    vector = (
                        fwd * translation[0]
                        + right * translation[1]
                        + np.array([0.0, 0.0, translation[2]])
                    )
                    combined[key] = fused_clearance(self.sensors.current, pose, vector)
                effect["combined_clearance_m"] = combined[key]
            effect["movement_violations"] = movement_reasons(
                values, observation, effect.get("combined_clearance_m"), limits
            )
            effect["travel_effect"] = {
                axis: (
                    "stopped"
                    if values[channel] == 0
                    else "axis_reached_brake"
                    if abs(body_delta[axis]) < (0.1 if axis == "up" else 0.2)
                    else "toward_goal"
                    if values[channel] * body_delta[axis] > 0
                    else "away_from_goal"
                )
                for channel, axis in zip(CHANNELS[:3], ("forward", "right", "up"))
            }
            rate = effect["resulting_controls"]["yaw_rate_rps"]
            if abs(error) <= math.radians(8):
                effect["turn_effect"] = (
                    "hold_aligned_heading" if rate == 0 else "turn_away_from_alignment"
                )
            elif rate == 0:
                effect["turn_effect"] = "heading_stays_unaligned"
            else:
                effect["turn_effect"] = (
                    "turn_toward_heading" if rate * error > 0 else "turn_away_from_heading"
                )
            if name in detours:
                effect["temporary_goal"] = detours[name][0]
                effect["planned_route"] = detours[name]
                effect["route_obstacles"] = observed_obstacles(
                    detours[name], pose, self.sensors.current
                )
        state = {
            "sensors": observation,
            "current_controls": snapshot,
            "task": self.task,
            "control_limits": limits,
            "goal": goal,
            "goal_offset": np.round(delta, 2).tolist(),
            "at_position": bool(at_position),
            "body_goal_offset_m": body_delta,
            "required_heading_degrees": self.task.get("heading_degrees"),
            "goal_tolerance_m": {
                "horizontal": 0.25 if self.detour else self.config.goal_horizontal_tolerance_m,
                "vertical": self.config.goal_vertical_tolerance_m,
            },
            "horizontal_distance": round(float(np.linalg.norm(delta[:2])), 2),
            "goal_bearing_error_degrees": round(math.degrees(error), 2),
            "requested_heading_degrees": math.degrees(desired) if desired is not None else None,
            "active_detour": self.detour,
            "control_effects": effects,
            "goal_path_clearance_m": fused_clearance(self.sensors.current, pose, delta)
            if not at_position
            else None,
            "recent_detours": self.detour_history[-6:],
            "recent_front_views": self.front_views[-3:],
            "recent_actions": [
                {"choice": c.get("choice"), "accepted": c.get("accepted")}
                for c in self.calls[-6:]
                if c["role"] == "control"
            ],
        }
        state["door_passage_complete"] = passage_complete
        endpoint_gap = self.sensors.observed_gap_at(np.array(goal))
        state["goal_observed_endpoint_gap_m"] = endpoint_gap
        state["goal_endpoint_observed_occupied"] = endpoint_gap is not None and endpoint_gap < 0.08
        if self.mission_handoffs:
            progress = self.facts(observation)
            progress["target_in_inspection_range"] = bool(
                progress["target_visible"]
                and progress["target_distance"] <= 1.45
                and observation["room"] == progress["objective"].get("room")
            )
            progress["dock_in_range"] = bool(
                "dock" in progress["objective"]
                and np.linalg.norm(pose - progress["objective"]["dock"]) <= 0.40
            )
            state["mission_progress"] = progress
        criteria = {}
        for name, effect in effects.items():
            meaning = (
                "Keep ALL currently active controls, including any ongoing pan."
                if name == "keep_controls"
                else "Stop translation AND stop pan."
                if name == "brake"
                else "Apply " + json.dumps(effect["patch"]) + "; retain omitted channels."
            )
            criteria[name] = (
                meaning
                + " Turn effect: "
                + effect["turn_effect"]
                + ". Movement violations: "
                + json.dumps(effect["movement_violations"])
            )
            if name == "brake_and_replan":
                criteria[name] += (
                    " Explicitly hand this blocked/failed route back to the planner. Not for ordinary braking or turning; no mission objective is completed."
                )
            if name in detours:
                criteria[name] += (
                    " Stop and select this bypass route, preserving the offset through its second leg: "
                    + json.dumps(detours[name])
                )
                criteria[name] += (
                    " Observed route obstacles: "
                    + json.dumps(effect["route_obstacles"])
                    + ". Prefer a route without observed blocked legs; unseen space is not guaranteed clear."
                )
        choice, log = self.request("control", state, PILOT, criteria)
        accepted = (
            choice is not None
            and self.world.status == "running"
            and self.world.time < self.config.seconds
            and log["latency_seconds"] <= self.world.lease
        )
        if accepted:
            receipt = self.world.update_controls(choice, choices[choice], snapshot["revision"])
            log["control_receipt"] = receipt
            accepted = receipt["applied"]
            if accepted:
                reasons = effects[choice]["movement_violations"]
                if reasons:
                    self.violations.append(
                        {"time": log["time"], "choice": choice, "reasons": reasons}
                    )
                if choice in detours:
                    self.detour = {
                        "position": detours[choice][0],
                        "remaining_route": detours[choice][1:],
                        "planned_route": detours[choice],
                        "leg": 0,
                        "time": self.world.time,
                        "selected_by": "jev",
                        "choice": choice,
                        "decision_time": log["time"],
                    }
                    self.last_progress = self.world.time
                elif choice == "brake_and_replan":
                    self.finish_for_replan(observation)
            else:
                log["rejection_reason"] = "control_state_changed_during_request"
        if not accepted:
            self.world.command("brake")
            if choice is not None:
                self.stale += 1
        log["accepted"] = accepted
        self.sensors.advance(max(0.0, 0.2 - log["latency_seconds"]))
