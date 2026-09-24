"""Host-side launcher for a frozen protocol, with one container per trial."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/protocols/native/protocol-recovery-followup.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = Path.cwd()
    output = args.out.resolve()
    relative = output.relative_to(root)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("control_schema") != "joystick-v1":
        raise ValueError(
            "Historical protocols require their frozen source; current controls use joystick-v1"
        )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    def run(job):
        case = job["case"]
        if not case.replace("_", "").isalnum():
            raise ValueError("Invalid case name")
        name = f"{case}-{int(job['seed'])}" + (
            "-" + job["prediction"] if "prediction" in job else ""
        )
        command = [
            "bash",
            str(root / "scripts/run_mission.sh"),
            "--out",
            str(relative / name),
            "--case",
            case,
            "--seed",
            str(job["seed"]),
            "--seconds",
            str(job["seconds"]),
            "--speed",
            str(protocol["speed_mps"]),
        ]
        command += ["--budget", str(protocol.get("budget_usd_per_trial", 0.45))]
        if protocol.get("find_only", False):
            command.append("--find-only")
        with (output / (name + ".console.log")).open("w") as log:
            process = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        return {"trial": name, "exit_code": process.returncode}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed = False
        for future in as_completed([pool.submit(run, job) for job in protocol["trials"]]):
            result = future.result()
            failed |= result["exit_code"] != 0
            print(json.dumps(result), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
