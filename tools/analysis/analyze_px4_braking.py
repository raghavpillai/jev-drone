"""Measure uninterrupted native braking segments; does not change flight policy."""

import bisect
import json
from pathlib import Path

import numpy as np


def measure(path):
    result = json.loads(path.read_text())
    trace, commands = result["trace"], result["commands"]
    times = [t["time"] for t in trace]
    rows = []
    for index, command in enumerate(commands):
        before, after = command["before"]["body_controls"], command["after"]["body_controls"]
        axes = ("forward_mps", "right_mps", "up_mps")
        if not any(before[a] for a in axes) or any(after[a] for a in axes):
            continue
        start_index = bisect.bisect_right(times, command["time"]) - 1
        if start_index < 0:
            continue
        initial = trace[start_index]
        lag = command["time"] - initial["time"]
        velocity = np.array(initial["velocity"])
        speed = float(np.linalg.norm(velocity))
        if lag > 0.1 or speed < 0.15:
            continue
        end_time = next(
            (
                c["time"]
                for c in commands[index + 1 :]
                if any(c["after"]["body_controls"][a] for a in axes)
            ),
            result["sim_seconds"],
        )
        stop_index = next(
            (
                i
                for i in range(start_index + 1, len(trace) - 1)
                if trace[i + 1]["time"] <= end_time
                and np.linalg.norm(trace[i]["velocity"]) <= 0.05
                and np.linalg.norm(trace[i + 1]["velocity"]) <= 0.05
            ),
            None,
        )
        if stop_index is None:
            rows.append(
                {
                    "trial": str(path.parent),
                    "time": command["time"],
                    "initial_speed_mps": speed,
                    "censored_by_next_motion_or_end": True,
                }
            )
            continue
        positions = np.array([t["position"] for t in trace[start_index : stop_index + 1]])
        along = (positions - positions[0]) @ (velocity / speed)
        observed_distance = float(max(along))
        rows.append(
            {
                "trial": str(path.parent),
                "time": command["time"],
                "initial_speed_mps": speed,
                "censored_by_next_motion_or_end": False,
                "trace_age_at_command_s": lag,
                "time_to_settle_sim_s": trace[stop_index]["time"] - command["time"],
                "max_along_track_displacement_m": observed_distance,
                "constant_0_5_acceleration_distance_m": speed**2,
                "effective_deceleration_mps2": speed**2 / (2 * observed_distance)
                if observed_distance > 0
                else None,
            }
        )
    return rows


def main():
    names = [
        "px4-recovery-depth-v15/controller-4910",
        "px4-recovery-cycle-v18/controller-4910",
        "px4-recovery-precision-v21/controller-4910",
        "px4-recovery-validation-v23/clutter-4931",
    ]
    rows = [r for name in names for r in measure(Path("results", name, "result.json"))]
    settled = [r for r in rows if not r["censored_by_next_motion_or_end"]]
    summary = {
        "zero_translation_transitions": len(rows),
        "settled_samples": len(settled),
        "censored_samples": len(rows) - len(settled),
        "speed_range_mps": [
            min(r["initial_speed_mps"] for r in settled),
            max(r["initial_speed_mps"] for r in settled),
        ],
        "maximum_observed_displacement_m": max(
            r["max_along_track_displacement_m"] for r in settled
        ),
        "samples_exceeding_constant_0_5_braking_term": sum(
            r["max_along_track_displacement_m"] > r["constant_0_5_acceleration_distance_m"]
            for r in settled
        ),
        "limitation": "Retrospective static indoor segments, sampled evaluator trajectory. Averages do not bound transient deceleration or validate a real-aircraft stopping envelope. Command/trace timing is approximate within 0.1 simulated seconds. Lease-expiration stops are not included.",
    }
    Path("report/px4-recovery/braking.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    speed = np.linspace(0.14, 0.29, 100)
    axis.plot(speed, speed**2, label="Constant 0.5 m/s² term", color="#b6573b")
    axis.plot(
        speed, speed**2 + 0.7 * speed, label="Term + 0.7 s settling allowance", color="#246b8e"
    )
    axis.scatter(
        [r["initial_speed_mps"] for r in settled],
        [r["max_along_track_displacement_m"] for r in settled],
        label="27 measured native stops",
        color="#222222",
        s=30,
        alpha=0.7,
    )
    axis.set(
        xlabel="Speed immediately before zero-translation command (m/s)",
        ylabel="Maximum travel along initial velocity (m)",
        title="PX4 braking component: observed displacement versus assumptions",
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", frameon=False)
    figure.text(
        0.1,
        0.015,
        "Reaction and depth margins omitted. Static slow-flight samples; one censored stop excluded.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig("report/px4-recovery/braking.png", dpi=160)
    figure.savefig("report/px4-recovery/braking.svg")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
