"""Infrastructure-only flight and sensor check, with no model calls."""
import argparse
import json
from pathlib import Path
import time

from jev_drone.world.world import Config
from jev_drone.sim.session import Session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    session = Session(Config(rig="surround"), args.out)
    try:
        session.launch()
        print("Connected", session.flight.position, session.flight.mode, flush=True)
        session.flight.takeoff(session.scene["start"])
        print("Hover", session.flight.position, session.flight.velocity, flush=True)
        time.sleep(3)
        result = {"position": session.flight.position.tolist(), "velocity": session.flight.velocity.tolist(),
                  "truth": session.transport.truth[1].tolist() if session.transport.truth else None,
                  "sim_time": session.transport.time,
                  "cameras": {name: {kind: [(t, a.shape) for t, a in list(items)[-1:]] for kind, items in channels.items()}
                              for name, channels in session.transport.images.items()},
                  "contacts": list(session.transport.contacts)[-3:], "messages": list(session.flight.messages)}
        (args.out/"smoke.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
