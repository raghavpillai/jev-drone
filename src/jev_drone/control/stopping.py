"""Declared experimental stopping assumptions, not a certified vehicle envelope."""


def required_clearance(requested, measured, frame_age, limits):
    if abs(requested) > 0.25:
        base = limits["full_clearance_m"]
    elif abs(requested) <= 0.1:
        base = limits.get("creep_clearance_m", limits["slow_clearance_m"])
    else:
        base = limits["slow_clearance_m"]
    if not limits.get("stopping_margin"):
        return base
    speed = max(abs(requested), abs(measured))
    # One delayed response plus one full lease after it; physics targets <=1x.
    horizon = (
        (frame_age if frame_age is not None else 0.65)
        + 2 * limits["command_lease_seconds"]
        + limits.get("braking_settling_allowance_s", 0.0)
    )
    # Native stops show settling beyond a constant-acceleration term. The added
    # allowance covers the sampled slow stops, not a certified full-speed bound.
    return max(base, speed * horizon + speed * speed / (2 * 0.5) + 0.08)
