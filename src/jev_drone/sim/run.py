"""Reproducible live Gazebo/PX4 trials, using OpenRouter Jev exclusively."""

import argparse
import json
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from jev_drone.control.policy import FrontExperiment
from jev_drone.evidence import snapshot_sources
from jev_drone.gateway import JevGateway, load_credential
from jev_drone.sim.backend import CameraSensors, World
from jev_drone.sim.body_check import run as body_check
from jev_drone.sim.config import Config
from jev_drone.sim.controller_check import ControllerCheck
from jev_drone.sim.session import Session


def freeze(output, config):
    hashes = snapshot_sources(output / "source")
    px4_commit = Path(".sim/PX4-Autopilot/.git/HEAD").read_text().strip()
    if len(px4_commit) != 40 or px4_commit.startswith("ref:"):
        raise ValueError("PX4 must be a detached, pinned source checkout")
    versions = {
        "px4": px4_commit,
        "gazebo": subprocess.check_output(["gz", "sim", "--versions"], text=True).strip(),
    }
    (output / "manifest.json").write_text(
        json.dumps({"sha256": hashes, "versions": versions}, indent=2)
    )
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))


def contact_check(world, sensors):
    deadline = time.monotonic() + 8
    while world.status == "running" and time.monotonic() < deadline:
        snapshot = world.flight.control_snapshot()
        world.update_controls("scripted_contact_check", {"forward_mps": -0.5}, snapshot["revision"])
        time.sleep(0.1)
    assert world.status == "collision" and world.contacts, (
        "Native contact sensor missed deliberate wall impact"
    )
    return {
        "status": "contact_verified",
        "world": world.metadata(),
        "trace": world.trace,
        "contacts": world.contacts,
        "sim_seconds": world.time,
        "wall_seconds": time.monotonic() - world.started_wall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", default="clutter")
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--budget", type=float, default=0.30)
    parser.add_argument(
        "--find-only",
        action="store_true",
        help="Finish after verifying the first item; omit return to launch",
    )
    parser.add_argument("--key-file", type=Path, default=Path("/run/secrets/openrouter.env"))
    parser.add_argument(
        "--calibrate", "--body-check", "--patch-check", dest="calibrate", action="store_true"
    )
    parser.add_argument("--contact-check", action="store_true")
    parser.add_argument("--controller-check", action="store_true")
    parser.add_argument("--force-newtons", type=float, default=0.0)
    args = parser.parse_args()
    config = Config(
        case=args.case,
        seed=args.seed,
        seconds=args.seconds,
        speed=args.speed,
        budget=args.budget,
        policy="v4",
        rig="front",
        wind=0,
        odometry_drift=0,
        servo="px4",
        disturbance_newtons=args.force_newtons,
        find_only=args.find_only,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    freeze(args.out, config)
    session = Session(config, args.out)
    gateway = world = None
    try:
        session.launch()
        session.flight.takeoff(session.scene["start"])
        print(
            f"PX4 airborne, mode={session.flight.mode}, armed={session.flight.armed}; starting "
            + ("calibration" if args.calibrate or args.contact_check else "Jev mission"),
            flush=True,
        )
        world = World(config, session)
        sensors = CameraSensors(world)
        if args.calibrate:
            result = body_check(world, sensors, args.out)
        elif args.contact_check:
            result = contact_check(world, sensors)
        else:
            gateway = JevGateway(
                "openrouter",
                load_credential("openrouter", args.key_file),
                journal=args.out / "calls.jsonl",
            )
            experiment = ControllerCheck if args.controller_check else FrontExperiment
            result = experiment(
                config, gateway, args.out / "result-frames", world=world, sensors=sensors
            ).run()
        result.update(
            id=args.out.name,
            setpoints=session.flight.sent,
            commands=world.commands,
            command_expirations=session.flight.expirations,
            pose_sync_rejections=sensors.pose_sync_rejections,
            realtime_factor=world.time / (time.monotonic() - world.started_wall),
        )
        (args.out / "result.json").write_text(json.dumps(result))
        print(
            json.dumps(
                {k: result[k] for k in ("status", "sim_seconds", "wall_seconds", "realtime_factor")}
            ),
            flush=True,
        )
    except Exception:
        (args.out / "failure.txt").write_text(traceback.format_exc())
        if session.flight:
            (args.out / "failure-flight.json").write_text(
                json.dumps(
                    {
                        "mode": session.flight.mode,
                        "armed": session.flight.armed,
                        "messages": list(session.flight.messages),
                        "setpoints": session.flight.sent,
                        "trace": world.trace if world else [],
                        "world_status": world.status if world else None,
                        "raw_contacts": list(session.transport.contacts),
                        "motion_error": session.motion_error,
                    }
                )
            )
        raise
    finally:
        if world:
            world.close()
        if gateway:
            gateway.close()
        session.close()


if __name__ == "__main__":
    main()
