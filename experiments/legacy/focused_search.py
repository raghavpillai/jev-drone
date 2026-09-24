"""Separate Jev's stopping decision from choosing the next search/inspection task."""

from experiments.legacy.purposeful_search import PurposefulSearch
from experiments.legacy.search_world import room_at
from experiments.legacy.sim import norm, subtract, unit

CONFIRMATION_DISTANCE = 2.0
APPROACH_STANDOFF = CONFIRMATION_DISTANCE - 0.65 - 0.1

STOPPING = """Decide whether to CONTINUE this object search or issue a terminal report.
Continue is one option covering ALL useful future actions: navigating to the requested room, looking around, moving to uninspected viewpoints, approaching objects and inspecting ambiguous appearances. Continue whenever any of these is useful.
Report found only for a clearly matching requested object in the requested room that is currently visible, within the confirmation distance of its observed surface, and with the drone stopped. Closed-door cabinets do not establish bookshelves. Apply the actual requested object's description.
Report search incomplete only when time is nearly exhausted or the remaining useful views have repeatedly failed. Seeing unrelated objects does not establish absence. An interrupted task is NOT a completed view; finish the remaining views. If uncompleted viewpoints remain with no failed attempts and substantial time remains, continue the search. Looking from high up does not see the entire room: distance and occlusion still matter. Never infer target absence from limited views.
"""

CONFIRMATION_RULES = """
For a found report, physical_confirmation_ready MUST be true. False means continue to approach, stop or look first, even if you recognized the target.
During final_budget_review, do not start another exploration pass. If a clearly matching target is visible_now and within_confirmation_distance but not stopped, continue_search to brake and finish confirmation within the remaining seconds. If physical_confirmation_ready is true, report the matching target. Otherwise report search incomplete. Being near and needing only to stop is a reason to finish confirmation, not abandon the task.
"""


class FocusedSearch(PurposefulSearch):
    api_error_limit = 3
    planning_instructions = (
        PurposefulSearch.planning_instructions.replace("2.15", str(CONFIRMATION_DISTANCE))
        + """
Use physical_confirmation_ready on the observed object. When a matching object is clearly visible but beyond the confirmation distance, APPROACH it. Repeating inspect from the same position cannot make it closer. If an approach fails because an obstacle blocks it, choose a new position or height that changes the viewing line, then approach again. Inspect is useful for ambiguous appearances or aiming toward a remembered object, not as a substitute for moving closer to a clearly visible target.
"""
    )

    def __init__(self, args, gateway):
        super().__init__(args, gateway)
        self.sim.command_lease = 0.5
        self.deadline_reviewed = False
        self.pilot.allow_slow = getattr(args, "slow", True)

    def options(self):
        options = super().options()
        for name, option in list(options.items()):
            if name.startswith("go_search_") and option["xyz"][2] > 2:
                option["xyz"] = [*option["xyz"][:2], 3.0]
                option["altitude_tolerance"] = 0.25
                option["description"] = (
                    option["description"].replace("2.8", "3.0")
                    + " Reach within 0.25m of the requested altitude before inspecting."
                )
                if self.args.memory == "coverage":
                    option["potential_unseen_by_look"] = self.attention.novelty(
                        option["xyz"], option["room"]
                    )
            if option["kind"] == "approach":
                point = self.memory[option["object"]]["visible_surface_xyz"]
                direction = unit(subtract(self.sim.position, point))
                # Current observed surface, with margin for the 0.65m arrival tolerance.
                option["xyz"] = [p + APPROACH_STANDOFF * d for p, d in zip(point, direction)]
                if getattr(self.args, "high_approach", True):
                    high_name = "approach_high_" + option["object"]
                    attempts = [t for t in self.tasks if t["choice"] == high_name]
                    options[high_name] = {
                        **option,
                        "xyz": [*option["xyz"][:2], 3.0],
                        "altitude_tolerance": 0.25,
                        "description": "Approach this observed object from above at 3.0m. Useful when the direct standoff is behind a lower occluder. "
                        + self.memory[option["object"]]["appearance"],
                        "failed_attempts": sum(
                            t["outcome"].startswith("stalled") for t in attempts
                        ),
                        "last_outcome": attempts[-1]["outcome"] if attempts else None,
                    }
        return options

    def confirmation_facts(self, obj):
        distance = norm(subtract(obj["visible_surface_xyz"], self.sim.position))
        visible = obj["id"] in {o["id"] for o in self.visible}
        stopped = norm(self.sim.velocity) <= 0.15
        return {
            "distance_to_observed_surface": round(distance, 2),
            "visible_now": visible,
            "within_confirmation_distance": distance <= CONFIRMATION_DISTANCE,
            "stopped": stopped,
            "physical_confirmation_ready": visible
            and distance <= CONFIRMATION_DISTANCE
            and stopped,
        }

    def state(self, options):
        state = super().state(options)
        state["confirmation_distance_m"] = CONFIRMATION_DISTANCE
        for obj in state["observed_objects"]:
            obj.update(self.confirmation_facts(obj))
        return state

    def stopping_state(self, options):
        return {
            "mission": self.scene.mission,
            "current_room": room_at(self.sim.position),
            "seconds_remaining": round(self.args.seconds - self.sim.time, 1),
            "speed_mps": round(norm(self.sim.velocity), 2),
            "final_budget_review": self.args.seconds - self.sim.time <= 8,
            "confirmation_distance_m": CONFIRMATION_DISTANCE,
            "objects": [
                {
                    "id": o["id"],
                    "appearance": o["appearance"],
                    "room": o["room"],
                    **self.confirmation_facts(o),
                }
                for o in self.memory.values()
                if o["room"] in self.scene.scope
            ],
            "remaining_view_goals": [
                name
                for name, o in options.items()
                if name.startswith("go_search_")
                and not o["completed_views"]
                and not o["failed_attempts"]
            ],
            "view_outcomes": [
                {k: o[k] for k in ("description", "completed_views", "failed_attempts")}
                for name, o in options.items()
                if name.startswith("go_search_")
            ],
            "recent_tasks": self.tasks[-5:],
        }

    def select_task(self, options):
        if self.args.seconds - self.sim.time <= 8:
            self.deadline_reviewed = True
        terminal = {name: o["description"] for name, o in options.items() if o["kind"] == "report"}
        for name in terminal:
            if name.startswith("report_found_"):
                obj = self.memory[name.removeprefix("report_found_")]
                ready = self.confirmation_facts(obj)["physical_confirmation_ready"]
                terminal[name] += (
                    f" Physical confirmation ready: {ready}. ONLY finish if True AND this is the requested object. If False, continue to approach or inspect first."
                )
        decision = self.request_choice(
            "mission",
            self.stopping_state(options),
            STOPPING + CONFIRMATION_RULES,
            {
                "continue_search": "Continue: choose a useful navigation, viewing, inspection or approach task.",
                **terminal,
            },
        )
        if (
            decision != "continue_search"
            or self.sim.status != "running"
            or self.sim.time >= self.args.seconds
        ):
            return decision
        tasks = {name: o for name, o in options.items() if o["kind"] != "report"}
        return self.request_choice(
            "planner",
            self.state(tasks),
            self.planning_instructions,
            {name: o["description"] for name, o in tasks.items()},
        )

    def task_outcome(self):
        if not self.deadline_reviewed and self.args.seconds - self.sim.time <= 8:
            return "time_budget_review"
        return super().task_outcome()

    def run(self):
        result = super().run()
        result["experiment"] = "focused-v15"
        return result
