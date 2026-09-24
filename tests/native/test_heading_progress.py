from collections import Counter
import math
from types import SimpleNamespace

from jev_drone.sim.config import Config
from jev_drone.control.policy import FrontExperiment


def experiment():
    agent = object.__new__(FrontExperiment)
    agent.config = Config()
    agent.mission_handoffs = False
    agent.world = SimpleNamespace(time=0.)
    agent.task = {'name':'view_test','position':[4.,0.,1.3],'look_position':[0.,0.,1.3]}
    agent.tasks = [dict(agent.task)]
    agent.detour = None
    agent.visits = Counter()
    agent.task_started = agent.last_progress = 0.
    agent.best_distance = 2.
    agent.best_heading_error = math.inf
    agent.progress_heading_phase = None
    return agent


def observation(yaw, position=(2.,0.,1.3), rate=.2):
    return dict(position_estimate=list(position),yaw_degrees=yaw,speed=0.,yaw_rate_rps=rate)


def test_turning_toward_travel_heading_is_progress_even_away_from_final_look():
    agent=experiment()
    for time,yaw in ((0,180),(6,135),(13,90),(20,45)):
        agent.world.time=time
        assert not agent.task_finished(observation(yaw))
    assert agent.last_progress == 20


def test_arrival_starts_a_separate_final_heading_progress_phase():
    agent=experiment()
    assert not agent.task_finished(observation(0))
    for time,yaw in ((8,0),(15,45),(22,90),(29,135)):
        agent.world.time=time
        assert not agent.task_finished(observation(yaw,position=(4.,0.,1.3)))
    agent.world.time=36
    assert agent.task_finished(observation(180,position=(4.,0.,1.3),rate=0.))
    assert agent.tasks[-1]['outcome']=='arrived'


def test_no_progress_still_stalls_and_total_task_limit_still_applies():
    agent=experiment()
    assert not agent.task_finished(observation(180))
    agent.world.time=13
    assert agent.task_finished(observation(180))
    assert agent.tasks[-1]['outcome']=='stalled'
    agent=experiment();agent.world.time=91
    assert agent.task_finished(observation(90))
    assert agent.tasks[-1]['outcome']=='stalled'


def test_altitude_only_goal_has_no_tiny_offset_bearing_or_premature_final_heading():
    agent=experiment();agent.task['position']=[2.,0.,2.2]
    obs=observation(90,position=(2.1,.05,1.3))
    assert agent.desired_heading(obs,agent.task['position'],False)==(None,'altitude')
    assert not agent.task_finished(obs)
    agent.world.time=8
    raised=observation(90,position=(2.1,.05,2.2))
    assert not agent.task_finished(raised)
    assert agent.progress_heading_phase==('waypoint','final_heading')
    assert agent.last_progress==8
    agent.task['heading_degrees']=90.
    assert agent.desired_heading(obs,agent.task['position'],False)==(math.pi/2,'fixed_heading')
