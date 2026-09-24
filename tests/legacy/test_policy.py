import httpx

from experiments.legacy.policy import Jev, sensor_text
from experiments.legacy.sim import ACTIONS, Simulation, scenario


def test_transport_preserves_live_answer_and_cost():
    controller = Jev("test-key")
    controller.client.close()
    answer = {"choice": "left", "confidence": .8,
              "probabilities": {a: float(a == "left") for a in ACTIONS}}
    controller.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "answers": {"movement": answer, "intent": {"choice": "pass_left"}},
        "model": "test-version", "usage": {"cost": .0001}})))
    try:
        result = controller.decide(Simulation(scenario("pillar", 0)).observation([]))
        assert result["action"] == "left"
        assert result["intent"] == "pass_left"
        assert result["usage"]["cost"] == .0001
    finally:
        controller.close()


def test_api_error_does_not_retry_or_expose_response_contents():
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(429, text="arbitrary server body")
    controller = Jev("test-key")
    controller.client.close()
    controller.client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        result = controller.decide(Simulation(scenario("open", 0)).observation([]))
        assert result["action"] == "brake"
        assert result["error"] == "http_429"
        assert len(calls) == 1
        assert "arbitrary" not in str(result)
    finally:
        controller.close()


def test_sensor_description_retains_blocked_actions_for_model_to_choose():
    sim = Simulation(scenario("pillar", 0))
    sim.position = (4.1, 5., 1.)
    text = sensor_text(sim.observation([]))
    assert "forward: BLOCKED" in text
    assert "left: OPEN" in text
    assert "forward" in ACTIONS


def test_simple_controls_are_seven_commands_without_changing_other_clients():
    simple, full = Jev("test", simple_controls=True), Jev("test")
    try:
        assert set(simple.criteria) == {"forward", "back", "left", "right", "up", "down", "brake"}
        assert len(full.criteria) == 11
    finally:
        simple.close()
        full.close()
