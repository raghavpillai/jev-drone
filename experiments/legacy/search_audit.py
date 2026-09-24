"""Reconstruct observed coverage and repeated search behavior from existing traces."""

import json
from collections import Counter
from pathlib import Path

from experiments.legacy.search_attention import SearchAttention, visible_probes
from experiments.legacy.search_world import room_at
from experiments.legacy.sim import Box, Scenario


def audit(path):
    d = json.loads(path.read_text())
    raw = d["world"]
    world = Scenario(
        raw["name"],
        raw["seed"],
        tuple(raw["start"]),
        tuple(raw["goal"]),
        tuple(Box(tuple(b["lo"]), tuple(b["hi"])) for b in raw["obstacles"]),
        tuple(raw["size"]),
    )
    memory = SearchAttention()
    entries = []
    previous_room = None
    for c in d["calls"]:
        if c["role"] != "control":
            continue
        state = c["state"]
        if isinstance(state, str):
            state = json.loads(state.splitlines()[1])
        position = state["position_metres_xyz"]
        look = state["camera"]["look_direction"]
        room = room_at(position)
        if room != previous_room:
            entries.append({"room": room, "time": c["sent_at"]})
            previous_room = room
        memory.record(visible_probes(world, position, look), position, look, c["sent_at"], room)
    for entry in entries:
        views = [
            v
            for v in memory.views
            if v["room"] == entry["room"] and entry["time"] <= v["time"] <= entry["time"] + 6
        ]
        entry["first_6s_looks"] = sorted({v["look"] for v in views})
    nav = Counter(c.get("choice") for c in d["calls"] if c["role"] == "navigator")
    return {
        "source": str(path),
        "status": d["status"],
        "room_entries": entries,
        "checklist_checked": d.get("checked_views", []),
        "observed_coverage": {r: memory.summary(r) for r in ("living_room", "bedroom", "study")},
        "control_observations": len(memory.views),
        "zero_gain_repeat_views": sum(
            v["repeat_view"] and v["new_visible_samples"] == 0 for v in memory.views
        ),
        "repeated_waypoints": {k: v for k, v in nav.items() if v > 1},
        "views": memory.views,
    }


def main():
    names = ("scanning-near", "scanning-far", "inspection-near", "all-jev-history-absent")
    results = [audit(Path("results") / name / "episode.json") for name in names]
    Path("report/search-audit.json").write_text(json.dumps(results, indent=2))
    for d in results:
        print(
            json.dumps(
                {
                    "source": d["source"],
                    "status": d["status"],
                    "coverage": {r: s["fraction"] for r, s in d["observed_coverage"].items()},
                    "repeat_views": d["zero_gain_repeat_views"],
                    "observations": d["control_observations"],
                    "repeated_waypoints": d["repeated_waypoints"],
                }
            )
        )


if __name__ == "__main__":
    main()
