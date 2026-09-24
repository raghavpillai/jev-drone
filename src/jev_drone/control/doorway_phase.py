"""Describe unfinished doorway alignment before assessing the crossing path."""

INSTRUCTIONS = """
For a doorway task, read doorway_phase BEFORE interpreting goal_path_status.
FACE_OPENING means finish facing the required heading, then stop pan.
CENTER_IN_OPENING means hold that heading and correct the remaining centerline
and altitude errors using observed-clear strafe/up/down controls. Brake residual
motion before reversing. A blocked/unknown DIAGONAL path to the far-side waypoint
at this stage does not establish that the door opening is blocked. First align
in front of the opening; do not repeatedly replan the same door while an allowed
alignment movement remains. After centering, CROSS_OPENING means move through
with the fixed heading and reassess actual forward clearance. An observed door
obstruction or forbidden movement must still be respected. These phase facts
never override movement_violations. Every control choice is yours.
"""


def describe(state, criteria):
    alignment = state.get('doorway_alignment')
    if alignment is None or state.get('active_detour'):
        return state, criteria
    state = dict(state)
    heading_ready = abs(alignment['heading_error_degrees']) <= 8
    lateral_ready = abs(alignment['centerline_offset_m']) <= .20
    vertical_ready = abs(alignment['altitude_error_m']) <= state['goal_tolerance_m']['vertical']
    phase = 'FACE_OPENING' if not heading_ready else 'CENTER_IN_OPENING' if not (lateral_ready and vertical_ready) else 'CROSS_OPENING'
    state['doorway_phase'] = {'phase':phase, 'heading_aligned':heading_ready,
        'centerline_aligned':lateral_ready, 'altitude_aligned':vertical_ready,
        'diagonal_goal_path_is_not_the_aligned_crossing':phase != 'CROSS_OPENING'}
    return state, criteria
