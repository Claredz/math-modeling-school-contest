"""Angle normalization helpers."""

from __future__ import annotations

from math import pi


def wrap_to_pi(angle_rad: float) -> float:
    """Map an angle to the open-left, closed-right interval (-pi, pi]."""

    wrapped = (float(angle_rad) + pi) % (2 * pi) - pi
    if wrapped <= -pi + 1e-14:
        return pi
    return wrapped
