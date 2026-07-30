"""Inertial bomb flight and fixed-centre smoke-cloud dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def detonation_position(
    release_position_m: np.ndarray,
    uav_velocity_mps: np.ndarray,
    *,
    delay_s: float = 3.5,
) -> np.ndarray:
    if delay_s < 0:
        raise ValueError("detonation delay cannot be negative")
    release = np.asarray(release_position_m, dtype=float)
    velocity = np.asarray(uav_velocity_mps, dtype=float)
    if release.shape != (2,) or velocity.shape != (2,):
        raise ValueError("release position and velocity must be two-dimensional")
    return release + delay_s * velocity


@dataclass(frozen=True)
class SmokeCloud:
    burst_time_s: float
    burst_center_m: np.ndarray
    maximum_radius_m: float = 120.0
    hold_duration_s: float = 18.0
    decay_duration_s: float = 5.0

    def __post_init__(self) -> None:
        center = np.asarray(self.burst_center_m, dtype=float).copy()
        if center.shape != (2,):
            raise ValueError("smoke centre must be two-dimensional")
        if self.maximum_radius_m <= 0:
            raise ValueError("maximum smoke radius must be positive")
        if self.hold_duration_s < 0 or self.decay_duration_s <= 0:
            raise ValueError("smoke durations must be nonnegative with positive decay")
        object.__setattr__(self, "burst_center_m", center)

    @property
    def hold_end_time_s(self) -> float:
        return self.burst_time_s + self.hold_duration_s

    @property
    def failure_time_s(self) -> float:
        return self.hold_end_time_s + self.decay_duration_s

    @property
    def decay_rate_mps(self) -> float:
        return self.maximum_radius_m / self.decay_duration_s

    def center(self, time_s: float) -> np.ndarray:
        if time_s < self.burst_time_s:
            raise ValueError("smoke centre does not exist before burst")
        return self.burst_center_m.copy()

    def radius(self, time_s: float) -> float:
        age_s = time_s - self.burst_time_s
        if age_s < 0 or age_s > self.hold_duration_s + self.decay_duration_s:
            return 0.0
        if age_s <= self.hold_duration_s:
            return self.maximum_radius_m
        return max(
            0.0,
            self.maximum_radius_m
            - self.decay_rate_mps * (age_s - self.hold_duration_s),
        )
