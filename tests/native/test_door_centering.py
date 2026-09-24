from copy import deepcopy

import pytest

from experiments.representations.door_centering import describe


def state(heading, yaw, offset):
    effects = {}
    for name, speed in [('left', -.1), ('right', .1), ('stop', 0.)]:
        effects[name] = {'resulting_controls': {'right_mps': speed},
                         'travel_effect': {'right': 'axis_reached_brake'},
                         'movement_violations': ['insufficient_depth'] if name == 'right' else []}
    return {'doorway_phase': {'phase': 'CENTER_IN_OPENING', 'centerline_aligned': False},
            'doorway_alignment': {'required_heading_degrees': heading, 'centerline_offset_m': offset},
            'sensors': {'yaw_degrees': yaw}, 'control_effects': effects}


@pytest.mark.parametrize('heading,yaw,toward', [(90, 95, 'left'), (-90, -95, 'right'),
                                              (0, 5, 'right'), (180, 175, 'left')])
def test_centerline_reference_survives_far_waypoint_projection(heading, yaw, toward):
    original = state(heading, yaw, .206)
    before = deepcopy(original)
    updated, criteria = describe(original, dict.fromkeys(original['control_effects'], ''), local_reference=True)
    assert original == before
    assert abs(updated['doorway_centering']['strafe_to_centerline_m']) > .206
    assert updated['control_effects'][toward]['travel_effect']['right'] == 'toward_goal'
    assert updated['control_effects']['stop']['travel_effect']['right'] == 'stopped'
    assert criteria.keys() == original['control_effects'].keys()
    for name, effect in updated['control_effects'].items():
        assert effect['resulting_controls'] == original['control_effects'][name]['resulting_controls']
        assert effect['movement_violations'] == original['control_effects'][name]['movement_violations']
        assert effect['far_waypoint_right_effect'] == 'axis_reached_brake'


def test_vertical_only_alignment_stops_lateral_motion():
    original = state(90, 90, .05)
    original['doorway_phase']['centerline_aligned'] = True
    updated, _ = describe(original, dict.fromkeys(original['control_effects'], ''), local_reference=True)
    assert updated['control_effects']['left']['travel_effect']['right'] == 'axis_reached_brake'
    assert updated['control_effects']['right']['travel_effect']['right'] == 'axis_reached_brake'


def test_other_phases_bypasses_and_completed_passages_are_unchanged():
    for override in ({'doorway_phase': {'phase': 'FACE_OPENING'}},
                     {'doorway_phase': {'phase': 'CROSS_OPENING'}},
                     {'active_detour': {'position': [1, 2, 3]}},
                     {'at_position': True}, {'door_passage_complete': True}):
        original = {**state(90, 95, .206), **override}
        assert describe(original, {}) == (original, {})
