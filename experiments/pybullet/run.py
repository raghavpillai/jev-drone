"""Run frozen rendered-sensor trials using live OpenRouter Jev decisions."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
import json
from pathlib import Path

from jev_drone.evidence import snapshot_sources
from jev_drone.gateway import JevGateway, load_credential
from jev_drone.planning.agent import Experiment
from jev_drone.world.world import Config


def run_one(job, out, key_file):
    config = Config(**{k: v for k, v in job.items() if k in {f.name for f in fields(Config)}})
    gateway = JevGateway("openrouter", load_credential("openrouter", key_file), journal=out/(job["id"]+".calls.jsonl"))
    try:
        result = Experiment(config, gateway, out/(job["id"]+"-frames")).run()
    finally:
        gateway.close()
    result["id"] = job["id"]
    path = out/(job["id"]+".json")
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(result))
    pending.replace(path)
    return {k: result[k] for k in ("id", "status", "stage", "sim_seconds", "cost_usd", "min_clearance", "near_miss_seconds")} | {
        "calls": len(result["calls"]), "violations": len(result["violations"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    jobs = json.loads(args.protocol.read_text())
    assert len({j["id"] for j in jobs}) == len(jobs)
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = snapshot_sources(args.out / "source", experiments=True)
    (args.out/"protocol.json").write_text(json.dumps(jobs, indent=2))
    (args.out/"manifest.json").write_text(json.dumps({"sha256": hashes, "workers": args.workers, "pybullet": "3.2.7"}, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job, args.out, args.key_file): job for job in jobs}
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                job = futures[future]
                (args.out/(job["id"]+".failure.json")).write_text(json.dumps({"id": job["id"], "error": type(error).__name__, "detail": str(error)}))
                raise


if __name__ == "__main__":
    main()
