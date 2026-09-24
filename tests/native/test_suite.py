import json
import sys
from types import SimpleNamespace

import pytest

from jev_drone.sim import suite


@pytest.mark.parametrize("exit_codes,expected_exit", [([0, 0], None), ([1, 0], 1)])
def test_suite_finishes_all_trials_and_reports_native_startup_failure(
    tmp_path, monkeypatch, exit_codes, expected_exit
):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "control_schema": "joystick-v1",
                "speed_mps": 0.5,
                "trials": [
                    {"case": "clutter", "seed": 1, "seconds": 1},
                    {"case": "sequence", "seed": 2, "seconds": 1},
                ],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "suite",
            "--protocol",
            str(protocol),
            "--out",
            str(tmp_path / "results"),
            "--workers",
            "1",
        ],
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=exit_codes[len(calls) - 1])

    monkeypatch.setattr(suite.subprocess, "run", run)
    if expected_exit is None:
        suite.main()
    else:
        with pytest.raises(SystemExit) as failure:
            suite.main()
        assert failure.value.code == expected_exit
    assert len(calls) == 2
    assert (tmp_path / "results/protocol.json").exists()
    assert len(list((tmp_path / "results").glob("*.console.log"))) == 2
