from experiments.legacy.continuation_report import control_violations, stationary_task_streak


def test_long_stationary_replanning_is_distinguished_from_a_moving_search():
    stuck = [{"position": [1, 2, 1.4]} for _ in range(20)]
    moving = [{"position": [i, 2, 1.4]} for i in range(20)]
    scan = [{"position": [1, 2, 1.4]} for _ in range(6)] + [{"position": [2, 2, 1.4]}]
    assert stationary_task_streak(stuck) == 20
    assert stationary_task_streak(moving) == 1
    assert stationary_task_streak(scan) == 6


def test_depth_audit_respects_speed_but_rejects_unknown_or_short_clearance():
    import json

    common = {"goal": {"inside_arrival_radius": False}}

    def call(reading, speed):
        return {
            "role": "control",
            "accepted": True,
            "action": "up",
            "speed": speed,
            "state": "header\n" + json.dumps(common) + "\nup: " + reading,
        }

    valid = call("BLOCKED, clearance 0.6 m", 0.3)
    assert not control_violations([valid])
    invalid = [
        call("UNKNOWN", 0.3),
        call("BLOCKED, clearance 0.4 m", 0.3),
        call("BLOCKED, clearance 0.6 m", 1),
    ]
    assert control_violations(invalid) == {"translation_without_sufficient_depth": 3}
