"""Run one Jev-only object search with the latest tested task interface."""

import argparse
import json
from pathlib import Path

from experiments.legacy.search_trials import prepare_output, run_one
from jev_drone.gateway import load_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["near", "far", "occluded", "absent"], default="occluded")
    parser.add_argument("--seed", type=int, default=710)
    parser.add_argument(
        "--item", choices=["bookcase", "bed", "plant", "cabinet", "table"], default="bookcase"
    )
    parser.add_argument(
        "--room", choices=["living_room", "bedroom", "study", "anywhere"], default="bedroom"
    )
    parser.add_argument(
        "--variation",
        choices=["original", "mirror_x", "mirror_y", "mirror_xy", "east_wall", "west_wall"],
        default="original",
    )
    parser.add_argument("--screen-height", type=float, default=2.15)
    parser.add_argument("--seconds", type=float, default=150.0)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    key = load_credential("openrouter", args.key_file)
    job = {k: v for k, v in vars(args).items() if k not in ("key_file", "out")}
    job.update(id="episode", engine="focused", memory="checklist", pilot="direct")
    prepare_output(args.out, job)
    print(json.dumps(run_one(key, args.out, job)), flush=True)


if __name__ == "__main__":
    main()
