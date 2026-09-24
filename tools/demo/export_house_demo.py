"""Export compact recorded trajectories and evaluator geometry to the local viewer."""

import argparse
import json
from pathlib import Path

from jev_drone.sim.config import Config
from jev_drone.sim.scene import HFOV, Geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        required=True,
        help="Mission directories or result.json files to include",
    )
    parser.add_argument("--out", type=Path, default=Path("demo/data"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    index = []
    paths = [path / "result.json" if path.is_dir() else path for path in args.source]
    for path in paths:
        result = json.loads(path.read_text())
        if "course" in result:
            continue
        intents = []
        journal = path.with_name("calls.jsonl")
        recorded = [json.loads(line) for line in journal.read_text().splitlines()]
        for call, raw in zip(result["calls"], recorded):
            if call["role"] != "planner" or call.get("error"):
                continue
            state = raw["state"]
            choice = call["choice"]
            option = state["options"].get(choice, {})
            phase = (
                "return"
                if "dock" in state["objective"]
                else {
                    "change_room": "navigate",
                    "search": "search",
                    "approach": "approach",
                    "report": "report",
                }.get(state.get("selected_task_kind"), "navigate")
            )
            memory = state.get("search_memory", {})
            inward = sum(v.get("inward_view_observed", False) for v in memory.values())
            survey = (
                f"{inward} / {len(memory)} inward views"
                if any("inward_view_observed" in v for v in memory.values())
                else None
            )
            label = (
                f"Survey {state['sensors']['room']} at " + ", ".join(map(str, option["position"]))
                if choice.startswith("view_")
                else f"Enter {option.get('leads_to_room', 'next room')}"
                if choice.startswith("cross_")
                else choice.replace("_", " ")
            )
            intents.append(
                dict(
                    time=call.get("response_time", call["time"]),
                    phase=phase,
                    task=label,
                    survey=survey,
                )
            )
        name = path.parent.parent.name + "--" + path.parent.name
        payload = dict(
            id=result["id"],
            status=result["status"],
            stage=result["stage"],
            world={
                k: result["world"][k]
                for k in ("geometry", "rooms", "doors", "mission", "start", "hfov")
            },
            trace=[
                {k: t[k] for k in ("time", "position", "velocity", "camera")}
                for t in result["trace"]
            ],
            duration=result["sim_seconds"],
            contacts=len(result["contacts"]),
            violations=len(result["violations"]),
            reports=[
                dict(time=r["time"], stage=r["stage"], valid=r["valid"]) for r in result["reports"]
            ],
            decisions=[
                dict(
                    time=c.get("response_time", c["time"]),
                    choice=c.get("choice", "API error"),
                    role=c["role"],
                )
                for c in result["calls"]
            ],
            intents=intents,
            provenance={
                "source": str(path),
                "motion": "Recorded Gazebo positions; interpolated between samples",
                "rendering": "Display reconstruction, not input to Jev",
                "model": "Jev via OpenRouter",
            },
        )
        (out / (name + ".json")).write_text(json.dumps(payload, separators=(",", ":")))
        version = result["world"]["config"]["pilot_representation"].split("-")[1]
        diagnostic = result["world"]["config"]["case"].startswith("house_search")
        index.append(
            {
                "path": "data/" + name + ".json",
                "label": version
                + " / "
                + ("Room diagnostic: " if diagnostic else "")
                + path.parent.name
                + " · "
                + result["status"],
                "success": result["status"] == "success",
                "safety_failure": bool(result["contacts"] or result["violations"]),
                "diagnostic": diagnostic,
            }
        )
    # Preserve the caller's ordering, with room-only diagnostics after missions.
    index.sort(key=lambda row: row["diagnostic"])
    scene = Geometry(Config(case="house_sequence", seed=5103))
    preview = dict(
        id="eight-room-house",
        status="scene_preview",
        stage=0,
        duration=0,
        contacts=0,
        violations=0,
        world=dict(
            geometry=scene.geometry,
            rooms=scene.layout.rooms,
            doors=scene.layout.doors,
            mission=scene.mission,
            start=scene.start.tolist(),
            hfov=HFOV,
        ),
        trace=[dict(time=0, position=scene.start.tolist(), velocity=[0, 0, 0], camera=[1, 0, 0])],
        decisions=[],
        reports=[],
    )
    (out / "scene.json").write_text(json.dumps(preview, separators=(",", ":")))
    index.append({"path": "data/scene.json", "label": "Explore furnished house · scene preview"})
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"flights": len(index) - 1, "index": str(out / "index.json")}))


if __name__ == "__main__":
    main()
