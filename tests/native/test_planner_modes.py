from copy import deepcopy
from types import SimpleNamespace

from jev_drone.control.policy import FrontExperiment
from jev_drone.planning.search_memory import SearchMemory
from jev_drone.planning.return_memory import ReturnMemory
from jev_drone.planning.door_memory import DoorMemory
from jev_drone.sim.config import Config


def test_jev_mode_limits_next_task_family_and_records_both_decisions():
    recorded = []
    def choose(state, instructions, criteria):
        recorded.append((deepcopy(state), dict(criteria)))
        choice = "change_room" if len(recorded) == 1 else "cross_living_bedroom"
        return {"choice": choice, "latency_seconds": .2, "usage": {}, "state": deepcopy(state)}
    experiment = object.__new__(FrontExperiment)
    experiment.config = Config()
    experiment.world = SimpleNamespace(time=20., realtime=True)
    experiment.gateway = SimpleNamespace(choose=choose)
    experiment.calls = []
    experiment.cost = 0.
    experiment.error_streak = 0
    experiment.error_started = None
    experiment.front_views = []
    experiment.search_memory = SearchMemory()
    experiment.return_memory = ReturnMemory()
    experiment.door_memory = DoorMemory()
    state = {"objective": {"dock": [1.4,2.,1.3]}, "sensors": {"room": "bedroom"},
        "ACTIVE_OBJECTIVE": {"return_to_launch": [1.4,2.,1.3], "room": "living"},
        "options": {"cross_living_bedroom": {}, "transit_bedroom_0_0_2.9": {}, "give_up": {}}}
    choice, _ = experiment.request("planner", state, "Plan.", {k:"Task." for k in state["options"]})
    assert choice == "cross_living_bedroom"
    assert [c["role"] for c in experiment.calls] == ["planner_mode", "planner"]
    assert set(recorded[1][1]) == {"cross_living_bedroom", "give_up"}
    assert 'search' not in recorded[0][1] and 'approach' not in recorded[0][1]
    assert recorded[0][0]['return_progress']['phase']=='RETURN_TO_LAUNCH_ROOM'
    assert recorded[1][0]["selected_task_kind"] == "change_room"
    assert "selected_task_kind" not in recorded[0][0]
    assert "selected_task_kind" not in state
