"""Presentation-only control experiments; candidate commands remain unchanged."""

INSTRUCTIONS = """
Read control_feasibility first. permitted_by_clearance lists commands with no
current movement-rule violations; it is not a route or a recommendation.
A direction that reduces the goal error can still be forbidden. If full/slow
motion is forbidden, check whether creep in that direction is permitted. If
no progress command is permitted, brake and observe or replan; never choose a
forbidden command just to satisfy a waypoint. All choices remain yours.
"""


def describe(state, criteria, *, summary=False):
    criteria = dict(criteria)
    effects = state['control_effects']
    forbidden = {}
    for name, effect in effects.items():
        reasons = effect.get('movement_violations', [])
        if reasons:
            forbidden[name] = list(reasons)
            criteria[name] = 'FORBIDDEN in this observation: ' + ', '.join(reasons) + '. This command cannot be used to make progress now.'
    if summary:
        state = dict(state)
        state['control_feasibility'] = {
            'permitted_by_clearance': [name for name, e in effects.items() if not e.get('movement_violations')],
            'forbidden': forbidden,
            'meaning': 'Observed movement constraints only. A permitted command may move away from the goal or enter unobserved space later.',
        }
    return state, criteria
