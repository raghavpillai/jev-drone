"""Source snapshots must remain importable outside the original checkout."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from jev_drone.evidence import snapshot_sources


def test_snapshot_can_replay_imports_without_the_checkout(tmp_path):
    destination = tmp_path / "source"
    hashes = snapshot_sources(destination, experiments=True)
    assert "jev_drone/control/controls.py" in hashes
    assert "experiments/legacy/policy.py" in hashes
    assert all("__pycache__" not in name for name in hashes)
    assert all(hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items())
    result = subprocess.run(
        [sys.executable, "-c", "from jev_drone.control import controls; "
         "from experiments.legacy import policy; print(controls.__file__); print(policy.__file__)"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(destination)},
        text=True, capture_output=True, check=True,
    )
    assert all(Path(line).is_relative_to(destination) for line in result.stdout.splitlines())
