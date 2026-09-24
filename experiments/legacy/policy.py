"""OpenRouter decision transport and observation-only comparison policies."""

import os
from pathlib import Path
import random
import re
import time

import httpx

from experiments.legacy.sim import ACTIONS, unit

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
PROMPT_VERSION = "v3-memory"
INSTRUCTIONS = """Which immediate movement should the drone make?
First avoid obstacles. Do not move in a direction whose sensor says BLOCKED, even when the destination is in that direction.
Then work toward the destination. When the direct way is blocked, choose an OPEN detour. It is fine to move away from the destination to escape a dead end.
Use recent positions to avoid repeating movements that make no progress. Keep a detour long enough to pass the obstruction.
The persistent maneuver is the drone's own previous decision. Follow it while the obstacle still blocks the route:
pass_left means keep moving left around it; pass_right means keep moving right; pass_above means climb; retreat means back out of the dead end.
Do not reverse a detour merely because the destination is now on the opposite side. Complete the maneuver before heading back toward the destination.
If already inside the arrival area, brake and stop. There is no automatic collision avoidance."""
CRITERIA = {name: f"Move {name.replace('_', '-')} at 1 m/s if that direction is OPEN and helps reach the destination or detour around an obstacle. Do not choose if BLOCKED."
            for name in ACTIONS if name != "brake"}
CRITERIA["brake"] = "Slow to a stop and hover: use when inside the arrival area, or when no movement is appropriate."
INTENT_INSTRUCTIONS = """Which maneuver should the drone remember for the next control cycle?
Maintain the persistent maneuver until the obstruction is passed. Do not alternate left and right just to point at the goal.
Choose direct when the path in the goal's direction is OPEN. If blocked, choose a detour through observed OPEN space.
If the current detour leads into a blocked direction, choose another maneuver. In a dead end with forward and both sides blocked, retreat.
This is a memory decision for future calls; the movement question in this call uses the previously stored maneuver."""
INTENT_CRITERIA = {
    "direct": "Move toward the destination when the way is open, or stop when arrived.",
    "pass_left": "Persist in passing the obstruction on its left until the goal direction becomes open.",
    "pass_right": "Persist in passing the obstruction on its right until the goal direction becomes open.",
    "pass_above": "Persist in climbing over the obstruction until the goal direction becomes open.",
    "retreat": "Back away from a dead end to find space for a detour.",
}


def sensor_text(observation: dict) -> str:
    """Plain-language sensor encoding; does not choose, filter, or rank actions.

    BLOCKED is a fixed 1.3 m measurement bucket, not a collision prediction.
    Both encodings expose the same range measurements and recent positions.
    """
    g = observation["goal"]
    lines = [
        "Fly to the destination and stop inside its 0.65 metre arrival radius.",
        observation["axes"],
        f"Destination is {g['direction']}, {g['distance_metres']} metres away. Inside arrival area: {g['inside_arrival_radius']}.",
        f"Destination offset [forward, left, up] in metres: {g['offset_metres_xyz']}.",
        f"Position: {observation['position_metres_xyz']}. Velocity [forward, left, up] m/s: {observation['velocity_metres_per_second_xyz']}.",
        f"Persistent maneuver from previous decision: {observation.get('persistent_intent', 'direct')}.",
        "Range sensors (vehicle size included; BLOCKED means obstacle or boundary within 1.3 metres):",
    ]
    for direction, reading in observation["ranges"].items():
        distance = reading["metres"]
        lines.append(f"{direction}: {'BLOCKED' if distance < 1.3 else 'OPEN'}, clearance {distance} metres.")
    lines += ["Ranges cap at 4 metres. Space beyond that and between rays is unknown.",
              "Movement speed 1 m/s; braking takes up to 0.5 seconds. The previous movement continues during an API call.",
              f"Recent observations and chosen movements, oldest first: {observation['recent']}"]
    return "\n".join(lines)


def load_key(path: Path | None = None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and path:
        match = re.search(r"(?m)^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*[\"']?([^\s\"'#]+)", path.read_text())
        key = match.group(1) if match else ""
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or pass --key-file; key values are never logged.")
    return key


class Jev:
    def __init__(self, key: str, dense_depth=False, long_memory=False, simple_controls=False):
        self.dense_depth = dense_depth
        self.long_memory = long_memory
        self.criteria = {name: description for name, description in CRITERIA.items() if not simple_controls or "_" not in name}
        self.client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=3., follow_redirects=False)

    def close(self):
        self.client.close()

    def decide(self, observation: dict) -> dict:
        state = sensor_text(observation)
        guidance = ""
        if self.dense_depth:
            state += "\nAdditional depth rays (direction: metres of vehicle clearance): " + str(observation["depth_rays"])
            guidance += "\nUse the upward diagonal depth rays to recognize space above a blocking obstacle. A horizontal goal can require climbing first. If horizontal detours remain obstructed and upward diagonal rays are clear, consider pass_above."
        if self.long_memory:
            state += "\nEarlier visited positions, goal distances, and movements (chronological): " + str(observation["breadcrumbs"])
            guidance += "\nUse the earlier visited positions to recognize repeated failed approaches. After escaping a dead end, continue around its outside instead of re-entering the same opening. Retreat means move away from the obstacle, which can be forward or back depending on the goal direction."
        payload = {"model": MODEL, "state": state, "questions": {
            "movement": {"type": "choice", "instructions": INSTRUCTIONS + guidance, "criteria": self.criteria},
            "intent": {"type": "choice", "instructions": INTENT_INSTRUCTIONS + guidance, "criteria": INTENT_CRITERIA}}}
        started = time.perf_counter()
        try:
            response = self.client.post(ENDPOINT, json=payload)
            elapsed = time.perf_counter()-started
            if response.status_code != 200:
                return {"action": "brake", "latency_seconds": elapsed, "error": f"http_{response.status_code}"}
            result = response.json()
            answer = result["answers"]["movement"]
            if answer["choice"] not in self.criteria:
                raise ValueError("Invalid action")
            probabilities = answer["probabilities"]
            if set(probabilities) != set(self.criteria) or any(not isinstance(p, (float, int)) or not 0 <= p <= 1 for p in probabilities.values()):
                raise ValueError("Invalid probabilities")
            confidence = answer["confidence"]
            if not isinstance(confidence, (float, int)) or not 0 <= confidence <= 1:
                raise ValueError("Invalid confidence")
            intent = result["answers"]["intent"]["choice"]
            if intent not in INTENT_CRITERIA:
                raise ValueError("Invalid intent")
            return {"action": answer["choice"], "latency_seconds": elapsed, "confidence": confidence,
                    "intent": intent, "intent_answer": result["answers"]["intent"],
                    "probabilities": probabilities, "model": result.get("model"),
                    "provider": result.get("provider"), "usage": result.get("usage", {})}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"action": "brake", "latency_seconds": time.perf_counter()-started, "error": "transport_or_response_error"}


def baseline(observation: dict, name: str, rng: random.Random) -> str:
    if name == "random":
        return rng.choice(list(ACTIONS))
    if observation["goal"]["inside_arrival_radius"]:
        return "brake"
    delta = unit(tuple(observation["goal"]["offset_metres_xyz"]))
    options = [action for action in ACTIONS if action != "brake"]
    if name == "reactive":
        # A deliberately small local rule; it never receives the hidden map.
        options = [a for a in options if observation["ranges"][a]["metres"] > 1.1]
    return max(options, key=lambda a: sum(x*y for x, y in zip(unit(ACTIONS[a]), delta)), default="brake")
