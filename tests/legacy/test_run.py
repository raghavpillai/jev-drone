from argparse import Namespace
import pytest

from experiments.legacy.run import episode, sample_history


class FixedController:
    def __init__(self, latency):
        self.latency = latency

    def decide(self, observation):
        return {"action": "forward", "intent": "direct", "latency_seconds": self.latency}


def args():
    return Namespace(speed=1., seconds=1., max_calls=10, timing="delayed", max_age=.8,
                     episode_budget=.025, policy="jev")


def test_response_is_applied_after_delay_and_old_command_keeps_moving():
    result = episode(args(), "open", 0, FixedController(.6))
    assert result["simulation_seconds"] == 1.
    assert result["decisions"][0]["received_at"] == pytest.approx(.6)
    assert result["decisions"][0]["accepted"]
    assert not result["decisions"][1]["accepted"]
    assert result["path_metres"] == pytest.approx(.16, abs=.001)


def test_stale_response_cannot_start_movement():
    result = episode(args(), "open", 0, FixedController(1.2))
    assert result["path_metres"] == 0
    assert result["stale_decisions"] == 1
    assert not result["decisions"][0]["accepted"]


def test_history_sampling_preserves_earliest_and_latest_observations():
    for length in [25, 47, 100, 150]:
        sampled = sample_history(list(range(length)))
        assert len(sampled) == 24
        assert sampled[0] == 0
        assert sampled[-1] == length-1
        assert sampled == sorted(set(sampled))
