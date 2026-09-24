"""Structured task handoffs and feedback. This module cannot command the vehicle."""

from experiments.legacy.search_world import ROOMS, room_at
from experiments.legacy.sim import norm, subtract

TASK_INSTRUCTIONS = """You are the mission planner. Assign a persistent task to the local Jev navigator.
It accepts search_room(room) or approach_object(observed_id), executes doorway navigation and local search, and reports progress. You never choose flight movements.
If the requested object has been identified, assign approach_object, then report found only after it is currently visible, within 2.1 m, and the drone is stopped.
Otherwise choose a relevant room with unchecked viewpoints. Honor a specifically requested room. For an unspecified room, use semantic likelihood and travel cost. Room names are priors, not proof of object presence.
Complete one room's search before needlessly repeating it. Coverage counts actual viewpoint visits, not attempts. All configured views checked means the finite search protocol for that room is exhausted, not a proof that every physical surface was seen.
After blocked feedback, try a different task or recover; do not blindly repeat a failed task. Once every relevant room has all views checked and no target has been identified, report_not_found. Do not report absence earlier.
Use only observed objects. Keep tasks narrow and persistent; the local agent chooses its intermediate waypoints."""

NAV_INSTRUCTIONS = """You are the local navigator executing assigned_task. Choose the next waypoint; another Jev call controls actual movement.
For search_room: enter that room, then visit its unchecked viewpoints. For approach_object: enter the object's observed room and approach its observed standoff point.
Rooms connect only through known doorways. First reach the approach on your current room's side, then cross to the other side. If it takes multiple room crossings, work through connected rooms.
Choose nearby unchecked viewpoints in the assigned room. Complete a selected waypoint before switching unless blocked or the target is discovered. Do not repeat completed search viewpoints while unchecked ones remain. A high viewpoint can see over low obstructions.
Use attempts and local_report to recover from stalls. You can report_blocked if you cannot make progress. Do not choose waypoints in unrelated rooms except to transit toward the assigned room.
Furniture locations are unknown; the controller uses its range sensors to avoid obstacles. Candidate points are not guaranteed reachable."""

WAYPOINT_INSTRUCTIONS = """You are the navigation/search planner. Choose the next waypoint or final report.
Find the requested object using only observations and memory. If recognized, approach its observed standoff point; report found only when currently visible within 2.1 m and stopped.
If not recognized, search unchecked viewpoints in relevant rooms. Honor a room specified in the mission. Otherwise use room semantics and travel distance, and complete one room's search before needlessly repeating it.
Cross walls only at doorways: first the approach on your current room's side, then the other side. Use connectivity for multiple crossings.
Keep an unfinished waypoint until arrival or stalled feedback. Do not repeat checked search points while unchecked ones remain. Try high views when low views are occluded.
If all configured viewpoints in every relevant room have actually been visited and no target is identified, report_not_found. This exhausts a finite search protocol, not a proof about every hidden surface. Never report absence before that condition.
You choose each waypoint. The Jev controller executes movement and avoidance. No hidden furniture map, target location, route search or automatic exploration is supplied."""


def room_coverage(scene, checked):
    return {
        room: {
            "checked": sum(
                name in checked for name in scene.points if name.startswith(f"search_{room}_")
            ),
            "total": sum(name.startswith(f"search_{room}_") for name in scene.points),
        }
        for room in ROOMS
    }


def all_checked(scene, checked):
    coverage = room_coverage(scene, checked)
    return all(coverage[r]["checked"] == coverage[r]["total"] for r in scene.scope)


def planning_state(scene, sim, memory, visible, checked, recognized, report):
    return {
        "mission": scene.mission,
        "current_room": room_at(sim.position),
        "position_xyz": [round(x, 2) for x in sim.position],
        "stopped": norm(sim.velocity) <= 0.15,
        "known_rooms": ROOMS,
        "known_doorways": scene.portals,
        "room_search_progress": room_coverage(scene, checked),
        "all_relevant_views_checked": all_checked(scene, checked),
        "observed_objects": [
            {
                "id": k,
                "appearance": v["appearance"],
                "room": v["room"],
                "observed_surface_xyz": v["visible_surface_xyz"],
            }
            for k, v in memory.items()
        ],
        "visible_now": visible,
        "recognized_target": recognized,
        "local_report": report,
    }


def task_options(memory):
    tasks = {
        f"search_{room}": {
            "kind": "search_room",
            "room": room,
            "objective": "Find the requested object from observations; inspect remaining viewpoints.",
            "complete_when": "target discovered or all configured room viewpoints checked",
            "report_on": ["target_discovered", "views_checked", "blocked"],
        }
        for room in ROOMS
    }
    for object_id, item in memory.items():
        tasks[f"approach_{object_id}"] = {
            "kind": "approach_object",
            "room": item["room"],
            "object_id": object_id,
            "objective": "Approach this observed object and stop nearby.",
            "complete_when": "stopped at the standoff waypoint",
            "report_on": ["arrived", "blocked"],
        }
    criteria = {k: str(v) for k, v in tasks.items()}
    criteria.update(report_options(memory))
    return tasks, criteria


def report_options(memory):
    return {
        "report_not_found": "End the search: all relevant configured views checked, no target identified.",
        **{
            f"report_found_{k}": f"Report {k} found: matches the mission, currently visible within 2.1 m, and stopped."
            for k in memory
        },
    }


def waypoint_options(scene, sim, memory, checked, attempts, task=None):
    points = {k: dict(v) for k, v in scene.points.items()}
    for object_id, obj in memory.items():
        points[f"approach_{object_id}"] = {
            "xyz": obj["approach_xyz"],
            "room": obj["room"],
            "description": f"Approach observed {object_id}: {obj['appearance']}",
        }
    if task:
        # Action scope comes only from the planner's task, never from hidden geometry.
        points = {
            k: v
            for k, v in points.items()
            if k.startswith("door_")
            or (task["kind"] == "search_room" and k.startswith(f"search_{task['room']}_"))
            or (task["kind"] == "approach_object" and k == f"approach_{task['object_id']}")
        }
    candidates = {
        k: {
            **v,
            "visited": k in checked,
            "attempts": attempts.count(k),
            "distance_metres": round(norm(subtract(v["xyz"], sim.position)), 2),
        }
        for k, v in points.items()
    }
    criteria = {k: v["description"] for k, v in points.items()}
    criteria.update(
        {
            "report_blocked": "The assigned task cannot make progress; ask the mission planner to revise it."
        }
        if task
        else report_options(memory)
    )
    return points, candidates, criteria
