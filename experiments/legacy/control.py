"""Jev-only velocity decisions with explicit dynamics and optional slow movement."""
from experiments.legacy.hierarchy import ask
from experiments.legacy.sim import ACTIONS, norm, subtract

MOVES = {f"{direction}_{pace}": (direction, speed)
         for direction in ("forward", "back", "left", "right", "up", "down")
         for pace, speed in (("cruise", 1.), ("slow", .4))}
MOVES["brake"] = ("brake", 0.)

CONTROL_INSTRUCTIONS = """Choose the drone's immediate velocity command. You are its only movement controller.
Reach the local waypoint and stop. World axes: forward +x, back -x, left +y, right -y, up +z, down -z.
Each command sets velocity on ONE axis and decelerates the other axes toward zero. Cruise is 1 m/s, slow is 0.4 m/s. Acceleration and braking are limited to 2 m/s². Inertia continues during inference (~0.35 s); changing direction does not instantly stop the old velocity.
Prefer cruise in clear space far from the waypoint; prefer slow for final alignment and near furniture. When moving toward an obstacle or about to overshoot, brake EARLY. predicted_braking_offset includes current inertia and an estimated 0.35 s reaction delay, not future acceleration. Use this only as a rough stopping cue.
Inside the arrival radius, brake until stopped. Do not keep correcting individual axes inside that radius. If an axis is aligned within 0.25 m, do not command movement on it unless an obstacle requires a detour; brake away existing drift instead. Keep the requested altitude unless detouring requires changing it.
Ranges measure clearance for the full drone. Space beyond 4 m and between sensor rays is unknown. A clearance below 0.8 m is tight. Never interpret a goal direction as permission to ignore an obstacle. No code will override a collision-causing choice.
If the direct direction is blocked, use the diagonal depth rays to choose a sustained side or vertical detour. Remember the recent path; do not oscillate repeatedly. If no useful movement is clear, brake."""


def control_state(sim, target, history, task):
    velocity = sim.velocity
    stop_time_half = norm(velocity)/(2*sim.acceleration)
    predicted = tuple(p+v*(.35+stop_time_half) for p,v in zip(sim.position, velocity))
    delta = subtract(target, sim.position)
    from experiments.legacy.search import depth_observation
    sensed = depth_observation(sim, [], target, "direct")
    ranges = {k: v["metres"] for k, v in sensed["ranges"].items()}
    ranges.update(sensed["depth_rays"])
    return {"local_task": task, "waypoint_xyz": target,
            "position_xyz": [round(v, 2) for v in sim.position],
            "velocity_xyz_mps": [round(v, 2) for v in velocity],
            "offset_to_waypoint_xyz": [round(v, 2) for v in delta],
            "predicted_braking_offset_xyz": [round(v, 2) for v in subtract(target, predicted)],
            "distance_to_waypoint_m": round(norm(delta), 2),
            "inside_arrival_radius": norm(delta) <= sim.goal_radius,
            "aligned_axes": [axis for axis, value in zip("xyz", delta) if abs(value) <= .25],
            "clearance_metres": ranges, "recent": history[-4:]}


def choose_control(client, sim, target, history, task, slow=True):
    moves = MOVES if slow else {k:v for k,v in MOVES.items() if not k.endswith("_slow")}
    criteria = {name: f"Set {direction} velocity to {speed} m/s; other axes slow toward zero."
                for name, (direction, speed) in moves.items()}
    criteria["brake"] = "Decelerate all movement to a hover. Use for arrival, braking before overshoot, or no clear movement."
    state = control_state(sim, target, history, task)
    result = ask(client, state, CONTROL_INSTRUCTIONS, criteria)
    action, speed = moves[result["choice"]]
    return {**result, "action": action, "speed": speed, "controller": "jev-dynamics-v1"}
