"""Interleaved latency probe: identical recorded states, 11 versus 7 controls."""
import argparse
import json
from pathlib import Path
import statistics
import time

from experiments.legacy.analyze import percentile
from experiments.legacy.policy import Jev, load_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists")
    episode = json.loads(Path("results/heldout-v3/pillar-1.json").read_text())
    states = [d["observation"] for d in episode["decisions"]][::4][:12]
    clients = {"11_actions": Jev(load_key(args.key_file)),
               "7_actions": Jev(load_key(args.key_file), simple_controls=True)}
    results = []
    try:
        for i, state in enumerate(states):
            order = list(clients) if i % 2 == 0 else list(reversed(clients))
            for name in order:
                started = time.perf_counter()
                answer = clients[name].decide(state)
                results.append({"variant": name, "sample": i, "total_seconds": time.perf_counter()-started, **answer})
    finally:
        for client in clients.values():
            client.close()
    summary = {}
    for name in clients:
        rows = [r for r in results if r["variant"] == name]
        durations = [r["total_seconds"] for r in rows]
        summary[name] = {"requests": len(rows), "median_ms": statistics.median(durations)*1000,
                         "p95_ms": percentile(durations,.95)*1000,
                         "serial_requests_per_second": len(rows)/sum(durations),
                         "errors": sum(bool(r.get("error")) for r in rows)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({"note":"Interleaved requests on repeated recorded states; includes first connection setup. Other experiments may overlap. Not a closed-loop success test.", "samples":results,"summary":summary},indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
