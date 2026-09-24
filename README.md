# Jev Drone

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) flies a simulated drone through an eight-room house to find a hidden item. It chooses where to go, where to look, and which movement controls to change. The simulation uses Gazebo and a PX4 X500 drone, with Jev calls through OpenRouter.

![Recorded Jev room search, with decisions and controls](demo/assets/preview.jpg)

This is a research prototype. Some missions succeed; others time out or fail safety checks. The demo includes both. No physical drone has been tested.

## Try the demo

Install Node.js/npm and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
npm ci --prefix demo
npm start --prefix demo
```

Open [localhost:8765](http://localhost:8765) to follow the drone, explore the house, or scrub through a flight. **Details** lets you switch recordings.

The [Jev decisions and controls view](http://localhost:8765/state.html) shows model inputs, returned choices, and applied controls at 12× playback. Press Space to pause or R to restart. Both views replay recorded flights; no API key or simulator setup is needed. The furniture styling and glowing target are viewer effects, separate from the sensor images.

## How it works

```text
RGB-D cameras → perception + memory → Jev planner → Jev controller → PX4 → drone
       ↑                                                                  │
       └──────────────────── new observations ────────────────────────────┘
```

1. Six simulated cameras capture RGB and depth at 160 × 120, 8 Hz. Code detects colored markers in the front camera and measures obstacle clearance using depth from all six directions. Jev receives these structured observations. Marker detection is handwritten; it cannot recognize arbitrary household objects.
2. Jev gets a pose estimate and a known map of rooms and doors. Furniture and target locations must be observed. Code remembers sightings, visited positions, viewing directions, and stalled tasks so Jev can use them in later decisions.
3. The Jev planner selects a task from supplied options: approach or cross a doorway, search from another position or height, look in a direction, approach a spotted target, or report an inspection. The chosen task and its progress go to the controller.
4. The Jev controller selects changes to forward/back, sideways, vertical, and turning controls. Controls persist between updates: stopping forward motion can leave a turn active. Jev chooses how to move toward the task and respond to obstacles.
5. PX4 stabilizes the drone and controls its motors. Movement commands expire after 0.9 seconds without renewal; stale telemetry also stops commanded movement.

Jev chooses from finite task and control options. Code prepares the observations, memory, and choices; there is no separate route solver or obstacle-avoidance autopilot. Localization uses simulated external odometry. This setup uses depth sensors and a room map, and does not test whether Jev can understand raw camera pixels.

Two find-only runs passed the strict audit with zero contacts or control-rule violations: entry → workshop in 268 simulated seconds, and the reverse in 920 seconds. Accepted control updates averaged 1.38–1.53 Hz in wall time. These two runs don't establish reliability. The featured 12× search uses an experimental v39 policy; v36 remains the default because later variants did not show more reliable full missions.

## Run a mission

Requires Linux, Docker, uv, and a multicore CPU. Setup builds PX4 v1.17.0 and Gazebo Harmonic; the first build takes time. uv uses the checked-in Python 3.10 version and dependency lockfile.

```bash
uv sync --locked
bash scripts/setup_simulator.sh
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
bash scripts/run_mission.sh \
  --case house_hidden --seed 5811 \
  --find-only --seconds 1200 --budget 3.5 \
  --out results/my-mission
```

The drone starts away from the workshop and must find its red marker. `--find-only` stops after a verified inspection; omit it to include the return objective. `house_reverse` starts at the other end of the house.

`--seconds` limits simulation time. `--budget` caps reported API cost in USD, checked after each call. Use a fresh output directory for each run. An exported `OPENROUTER_API_KEY` also works. Credentials and results are ignored by Git.

Each run saves model requests and responses, controls, trajectory, sensor frames, safety events, and a source snapshot. The bundled demo contains compact replays; full experiment archives stay local.

## Working on the code

Start with [control/policy.py](src/jev_drone/control/policy.py) and [planning/agent.py](src/jev_drone/planning/agent.py). Within [`src/jev_drone/`](src/jev_drone/), `perception/` processes sensor data, `sim/` runs PX4/Gazebo, `world/` defines the house, and `audit/` checks mission results. `experiments/` holds protocols and earlier prototypes; `tools/` handles analysis and replay exports.

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
```

Native tests skip when simulator dependencies are missing. Run the full suite in the simulator image:

```bash
docker run --rm -e PYTHONPATH=/workspace/src \
  -v "$PWD:/workspace" jev-px4:harmonic python3 -m pytest -q
```

Check flight controls without API calls:

```bash
bash scripts/run_mission.sh --body-check --out results/body-check
```

With the demo server running, check the browser UI:

```bash
uv run --locked --extra video playwright install chromium
uv run --locked --extra video python -m tests.browser.check_preview
```

## Export a replay or video

Export a completed mission and its Jev state. This replaces the demo index:

```bash
uv run --locked python -m tools.demo.export_house_demo --source results/my-mission
uv run --locked python -m tools.demo.export_state_replay \
  --source results/my-mission --out demo/data/state-replay.json
```

With Chromium installed as above and the demo server running, render the search at 1080p, 30 fps:

```bash
uv run --locked --extra video python -m tools.demo.render_video \
  --speed 12 --out report/my-mission.mp4
```

The renderer checks frame timing and duplicates, and saves a provenance manifest beside the MP4.
