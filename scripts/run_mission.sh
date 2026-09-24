#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mounts=(-v "$task_root:/workspace")
needs_key=true
for argument in "$@"; do
  case "$argument" in
    --calibrate|--contact-check|--patch-check|--body-check|--help|-h) needs_key=false ;;
  esac
done
if [[ "$needs_key" == true ]]; then
  if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    mounts+=(-e OPENROUTER_API_KEY)
  else
    credential_file="${JEV_KEY_FILE:-$task_root/.env}"
    [[ -r "$credential_file" ]] || { echo "Add OPENROUTER_API_KEY to .env, export it, or set JEV_KEY_FILE." >&2; exit 1; }
    credential_file="$(realpath "$credential_file")"
    mounts+=(-v "$credential_file:/run/secrets/openrouter.env:ro")
  fi
fi
exec docker run --rm --init --cpus 12 -e LP_NUM_THREADS=4 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/src "${mounts[@]}" \
  jev-px4:harmonic python3 -m jev_drone.sim.run "$@"
