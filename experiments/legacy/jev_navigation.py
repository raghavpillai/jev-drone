"""Jev movement, camera look and recovery decisions with optional observed-space memory."""
import json

from experiments.legacy.observed_map import ObservedMap, recovery_options
from experiments.legacy.policy import CRITERIA, INSTRUCTIONS, INTENT_CRITERIA, INTENT_INSTRUCTIONS
from experiments.legacy.representation_experiment import DIRECTIONS, GUIDANCE, encode, label, observe
from experiments.legacy.sim import ACTIONS, norm, subtract, unit

LOOK = {name: ACTIONS[name] for name in ("forward", "back", "left", "right", "up", "down")}
RECOVERY = """Choose a temporary recovery goal or resume the mission waypoint.
Use observed-space memory and previous failed attempts. A trap may require moving AWAY from the mission before going around it. Do not repeatedly choose goals that already failed. A previous position is evidence of visitation, not a guaranteed straight route.
Explore beyond the open end of a dead end, then around its outside before returning toward the mission. Current clearances are local only. You decide the recovery; no pathfinder or automatic escape exists.
"""


class Navigation:
    def __init__(self, gateway, mode="history", limited=False, degraded=False, seed=0):
        self.gateway, self.mode = gateway, mode
        self.limited, self.degraded, self.seed = limited, degraded, seed
        self.map = ObservedMap()
        self.look = "forward"
        self.look_history = [(0., "forward")]
        self.intent = "direct"
        self.history, self.events = [], []
        self.recovery_target = None
        self.target = None
        self.best = float("inf")
        self.last_progress = 0.
        self.last_recovery = -10.

    def observation(self, sim, target):
        common, readings = observe(sim, self.history, self.intent, self.degraded, self.seed, target=target)
        capture_time = sim.time-common["sensor"]["age_seconds"]
        capture_look = next((look for when,look in reversed(self.look_history) if when <= capture_time+1e-3), "forward")
        if self.limited:
            for direction in DIRECTIONS:
                if sum(a*b for a,b in zip(unit(direction),LOOK[capture_look])) < .5:
                    readings[label(direction)] = None
        if self.mode != "history":
            self.map.record(common["sensor"]["capture_position"], readings, sim.time-common["sensor"]["age_seconds"])
        common["camera"] = {"look_direction": self.look, "captured_look_direction": capture_look, "coverage": "120 degree cone" if self.limited else "surround",
                            "look_changes": "Jev may aim the camera independently of fixed-heading translation; no flight yaw modeled."}
        return common, readings

    def recovery(self, sim, target, report):
        if self.mode != "recovery" or sim.time-self.last_progress < 6 or sim.time-self.last_recovery < 6:
            return None
        common, readings = self.observation(sim,target)
        points, criteria = recovery_options(sim.position,self.map)
        state = {"mission_waypoint": target, "local_report": report, "current": encode(common,readings,"brief"),
                 "observed_map": self.map.describe(sim.position,sim.time), "previous_recovery_attempts": self.events[-8:]}
        result = self.gateway.choose(state,RECOVERY,criteria)
        self.last_recovery = sim.time
        if not result.get("error"):
            self.recovery_target = points[result["choice"]]
            self.events.append({"at":sim.time,"choice":result["choice"],"xyz":self.recovery_target})
            self.intent = "direct"
        return result

    def decide(self, sim, target, extra=None):
        if self.target != tuple(target):
            self.target = tuple(target)
            self.recovery_target = None
            self.best = norm(subtract(target,sim.position))
            self.last_progress = sim.time
            self.intent = "direct"
        if self.recovery_target and norm(subtract(self.recovery_target,sim.position)) <= .65:
            self.recovery_target = None
        active_target = self.recovery_target or target
        common, readings = self.observation(sim,active_target)
        common["assigned_task"] = extra or {}
        common["mission_waypoint_xyz"] = list(target)
        if self.mode != "history":
            common["observed_map"] = self.map.describe(sim.position,sim.time)
        state = encode(common,readings,"brief")
        result = self.gateway.evaluate(state,self.questions())
        return self.decode(result)

    def questions(self):
        questions = {
            "movement": {"type":"choice","instructions":INSTRUCTIONS+GUIDANCE,
                         "criteria":{k:v for k,v in CRITERIA.items() if "_" not in k}},
            "intent": {"type":"choice","instructions":INTENT_INSTRUCTIONS+GUIDANCE,"criteria":INTENT_CRITERIA},
        }
        if self.limited:
            questions["look"] = {"type":"choice","instructions":
                "Where should the camera look for the NEXT observation? Missing directions are unknown. Look toward intended travel or unexplored space. If stopped at a search viewpoint, inspect missing_views_here before leaving. Camera aim is independent of movement; it does not rotate the flight axes.",
                "criteria":{k:f"Aim the camera {k}; observe that cone next cycle." for k in LOOK}}
        return questions

    def decode(self,result):
        if not result.get("error"):
            result["action"] = result["answers"]["movement"]["choice"]
            result["intent"] = result["answers"]["intent"]["choice"]
            if self.limited:
                result["look"] = result["answers"]["look"]["choice"]
        else:
            result["action"] = "brake"
        return result

    def accept(self, sim, decision, accepted, target):
        if accepted:
            self.intent = decision["intent"]
            self.look = decision.get("look",self.look)
            self.look_history.append((sim.time,self.look))
        self.history.append({"position":[round(p,2) for p in sim.position],
                             "action":decision["action"] if accepted else "brake"})
        distance = norm(subtract(target,sim.position))
        if distance < self.best-.15:
            self.best, self.last_progress = distance, sim.time
