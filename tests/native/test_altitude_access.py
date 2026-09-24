from copy import deepcopy

from experiments.representations.altitude_access import describe
from experiments.representations.control_presentation import describe as presentation


def state():
    effects={name:{'movement_violations':['insufficient_or_unknown_down_depth'] if name.startswith('down') else [],
                   'resulting_controls':{'up_mps':-.1 if name=='down_creep' else -.25 if name=='down' else 0.}}
             for name in ('down','down_creep','up','up_creep','brake','brake_and_replan')}
    return {'body_goal_offset_m':{'up':-.3},'horizontal_distance':.1,
            'goal_tolerance_m':{'horizontal':.35,'vertical':.15},'at_position':False,'control_effects':effects}


def test_blocked_descent_does_not_mask_choices_or_change_commands():
    original=state();before=deepcopy(original);criteria=dict.fromkeys(original['control_effects'],'Original.')
    updated,choices=describe(original,criteria)
    assert original==before and set(choices)==set(criteria)
    assert updated['control_effects']==original['control_effects']
    assert updated['altitude_access']['direction']=='down'
    assert updated['altitude_access']['requested_vertical_motion_permitted'] is False
    assert updated['altitude_access']['horizontal_reached'] is True
    assert 'another approach' in choices['brake_and_replan']
    assert criteria['brake_and_replan']=='Original.'


def test_creep_remains_possible_when_faster_descent_is_forbidden():
    original=state();original['control_effects']['down_creep']['movement_violations']=[]
    updated,choices=describe(original,dict.fromkeys(original['control_effects'],'Original.'))
    assert updated['altitude_access']['permitted_vertical_controls']==['down_creep']
    assert updated['altitude_access']['requested_vertical_motion_permitted'] is True
    assert choices['brake_and_replan']=='Original.'


def test_no_altitude_correction_at_arrival_or_within_tolerance():
    for change in [{'at_position':True},{'body_goal_offset_m':{'up':-.15}},{'body_goal_offset_m':{'up':0}}]:
        original={**state(),**change}
        assert describe(original,dict.fromkeys(original['control_effects'],''))[0] is original


def test_ascent_uses_up_controls_and_preserves_sign():
    original=state();original['body_goal_offset_m']['up']=.4
    original['horizontal_distance']=2
    updated,_=describe(original,dict.fromkeys(original['control_effects'],''))
    assert updated['altitude_access']['direction']=='up'
    assert updated['altitude_access']['remaining_m']==.4
    assert updated['altitude_access']['permitted_vertical_controls']==['up','up_creep']
    assert updated['altitude_access']['horizontal_reached'] is False


def test_summary_is_only_a_different_presentation_of_existing_constraints():
    original=state();before=deepcopy(original);criteria=dict.fromkeys(original['control_effects'],'Original.')
    updated,choices=presentation(original,criteria,summary=True)
    assert set(choices)==set(criteria) and original==before
    assert updated['control_effects']==original['control_effects']
    assert 'down' in updated['control_feasibility']['forbidden']
    assert 'up' in updated['control_feasibility']['permitted_by_clearance']
    assert choices['down'].startswith('FORBIDDEN')
    assert choices['up']=='Original.'
