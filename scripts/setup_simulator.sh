#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
px4_source="$task_root/.sim/PX4-Autopilot"
px4_commit=d6f12ad1c4f70ad3230afd7d86e971421e02fef4
if [[ ! -d "$px4_source" ]]; then
  mkdir -p "$task_root/.sim"
  git clone --branch v1.17.0 --depth 1 https://github.com/PX4/PX4-Autopilot.git "$px4_source"
fi
if [[ "$(git -C "$px4_source" rev-parse HEAD)" != "$px4_commit" ]]; then
  echo "PX4 checkout differs from the tested commit; use a separate .sim directory." >&2
  exit 1
fi
git -C "$px4_source" checkout --detach "$px4_commit"
git -C "$px4_source" submodule update --init --recursive --depth 1 --jobs 8 \
  Tools/simulation/gz src/modules/mavlink/mavlink src/lib/events/libevents \
  src/lib/crypto/monocypher src/modules/uxrce_dds_client/Micro-XRCE-DDS-Client \
  src/lib/heatshrink/heatshrink src/drivers/gps/devices src/lib/cdrstream/cyclonedds src/lib/cdrstream/rosidl
docker build -t jev-px4:harmonic -f docker/Dockerfile docker
docker run --rm -v "$task_root:/workspace" -w /workspace/.sim/PX4-Autopilot \
  jev-px4:harmonic bash -lc 'git config --global --add safe.directory "*"; make px4_sitl_default -j16'
