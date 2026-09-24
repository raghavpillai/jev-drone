"""Atomic absolute control updates; callers hold the flight transport lock."""

import math

BODY_CHANNELS = ("forward_mps", "right_mps", "up_mps", "yaw_rate_rps")


def changed_channels(current, assignments):
    return {channel: value for channel, value in assignments.items() if current[channel] != value}


class ControlLatch:
    axes = ("x", "y", "z")

    def __init__(self, lease=0.6, *, axes=None, state_key="velocity_mps"):
        self.lease = lease
        self.axes = tuple(axes) if axes is not None else self.axes
        self.state_key = state_key
        self.values = dict.fromkeys(self.axes, 0.0)
        self.revision = 0
        self.expires = None

    def snapshot(self):
        return {self.state_key: dict(self.values), "revision": self.revision}

    def expire(self, now, telemetry_stale=False):
        if self.expires is None or (now < self.expires and not telemetry_stale):
            return None
        before = self.snapshot()
        self.values = dict.fromkeys(self.axes, 0.0)
        self.revision += 1
        self.expires = None
        return {
            "before": before,
            "after": self.snapshot(),
            "reason": "telemetry_stale" if telemetry_stale else "lease_expired",
        }

    def update(self, patch, now, expected_revision=None):
        if any(
            axis not in self.axes
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or abs(value) > 1
            for axis, value in patch.items()
        ):
            raise ValueError("Unknown control channel or non-finite/value outside [-1, 1]")
        before = self.snapshot()
        applied = expected_revision is None or expected_revision == self.revision
        if applied:
            self.values.update(patch)
            self.revision += 1
            self.expires = now + self.lease
        return {
            "applied": applied,
            "patch": dict(patch),
            "expected_revision": expected_revision,
            "before": before,
            "after": self.snapshot(),
            "wall": now,
        }
