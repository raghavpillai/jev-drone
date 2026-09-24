from collections import Counter
from types import SimpleNamespace

from jev_drone.control.policy import FrontExperiment, movement_reasons, patches
from jev_drone.control.stopping import required_clearance


def test_creep_has_a_smaller_stopping_envelope_but_cannot_use_unknown_depth():
    limits = dict(creep_clearance_m=.4, slow_clearance_m=.7, full_clearance_m=1.5,
                  command_lease_seconds=.9, stopping_margin=True)
    observation = dict(body_clearance_m={"forward":.6}, body_velocity_mps={"forward":0.},
                       yaw_rate_rps=0., frame_age_seconds=.15)
    assert movement_reasons(patches(.5)["forward_slow"], observation, limits=limits)
    assert not movement_reasons(patches(.5)["forward_creep"], observation, limits=limits)
    observation["body_clearance_m"]["forward"] = None
    assert movement_reasons(patches(.5)["forward_creep"], observation, limits=limits)
    assert required_clearance(.1, .1, .65, limits) <= .4


def test_replan_brakes_every_axis_and_preserves_the_mission_stage():
    assert not any(patches(.5)["brake_and_replan"].values())
    experiment = object.__new__(FrontExperiment)
    experiment.world = SimpleNamespace(time=7.)
    experiment.stage = 1
    experiment.task = {"name":"cross_living_bedroom"}
    experiment.tasks = [{**experiment.task, "outcome":"active"}]
    experiment.detour = {"position":[5.,2.5,1.3]}
    experiment.visits = Counter()
    experiment.finish_for_replan({"position_estimate":[6.8,2.4,1.3]})
    assert experiment.stage == 1
    assert experiment.task is None and experiment.detour is None
    assert experiment.tasks[-1]["outcome"] == "replan_requested"
    assert experiment.visits["cross_living_bedroom"] == 1
