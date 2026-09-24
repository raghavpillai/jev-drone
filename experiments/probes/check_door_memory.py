"""Paired Jev planning probes using doorway evidence from recorded RGB-D frames."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from jev_drone.gateway import JevGateway, load_credential
from jev_drone.perception.sensors import Frame, points
from jev_drone.planning.door_memory import DoorMemory, INSTRUCTIONS


def recorded_cloud(path, hfov):
    with np.load(path) as data:
        names = data["camera_names"].tolist()
        if "right" in data:
            right, up = data["right"], data["up"]
        else:
            # Older archives stored only forward. Synchronized orthogonal cameras
            # supply the missing basis exactly; reject asynchronous old snapshots.
            if np.ptp(data["times"]) > 1e-8 or set(names) != {"east", "west", "north", "south", "up", "down"}:
                raise ValueError("Old camera snapshot lacks a reconstructible synchronized basis")
            x, y, z = (data["forward"][names.index(n)] for n in ("east", "north", "up"))
            rights = {"east": -y, "west": y, "north": x, "south": -x, "up": -y, "down": -y}
            ups = {"east": z, "west": z, "north": z, "south": z, "up": -x, "down": x}
            right, up = [rights[n] for n in names], [ups[n] for n in names]
        clouds = []
        for index, name in enumerate(names):
            frame = Frame(float(data["times"][index]), data["rgb"][index], data["depth"][index],
                data["origins"][index], data["forward"][index], right[index], up[index],
                data["depth"].shape[2]/(2*math.tan(hfov/2)), name)
            cloud = points(frame)
            clouds.append(cloud[np.isfinite(cloud).all(axis=2)])
        return np.concatenate(clouds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gateway = JevGateway("openrouter", load_credential("openrouter", Path(".env")), journal=args.out/"calls.jsonl")
    rows = []
    try:
        for name in ("blocked_door-4875", "clutter-4871", "vertical-4874"):
            path = Path("results/px4-autonomy-modes-validation")/name
            result = json.loads((path/"result.json").read_text())
            call = next(c for c in result["calls"] if c["role"] == "planner" and not c.get("error"))
            snapshot = sorted((path/"result-frames").glob("*.npz"))[0]
            memory = DoorMemory()
            memory.observe(recorded_cloud(snapshot, result["world"]["hfov"]), float(snapshot.stem))
            for repeat in range(3):
                for variant in ("baseline", "observed_door_surface"):
                    question = call["questions"]["selection"]
                    state, criteria, instructions = call["state"], question["criteria"], question["instructions"]
                    if variant != "baseline":
                        state, criteria = memory.describe(state, criteria)
                        instructions += INSTRUCTIONS
                    response = gateway.choose(state, instructions, criteria)
                    row = dict(trial=name, frame=str(snapshot), repeat=repeat, variant=variant,
                        observed_obstructions=memory.obstructions, choice=response.get("choice"), error=response.get("error"),
                        latency_seconds=response["latency_seconds"])
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        gateway.close()
        (args.out/"results.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
