"""Continuous piecewise-linear paths for UAVs launched from the moving ship."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smoke_defense.dynamics import ShipMotion

UAV_SPEED_MPS = 28.0
POSITION_TOLERANCE_M = 1e-8
TIME_TOLERANCE_S = 1e-10


@dataclass(frozen=True)
class LinearFlightSegment:
    start_time_s: float
    end_time_s: float
    start_position_m: np.ndarray
    end_position_m: np.ndarray

    def __post_init__(self) -> None:
        if self.end_time_s <= self.start_time_s:
            raise ValueError("flight segment must have positive duration")
        start = np.asarray(self.start_position_m, dtype=float).copy()
        end = np.asarray(self.end_position_m, dtype=float).copy()
        if start.shape != (2,) or end.shape != (2,):
            raise ValueError("flight segment positions must be two-dimensional")
        object.__setattr__(self, "start_position_m", start)
        object.__setattr__(self, "end_position_m", end)
        if not np.isclose(
            self.speed_mps,
            UAV_SPEED_MPS,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("every airborne segment must have speed 28 m/s")

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def velocity_mps(self) -> np.ndarray:
        return (self.end_position_m - self.start_position_m) / self.duration_s

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_mps))

    def position(self, time_s: float) -> np.ndarray:
        if not self.start_time_s <= time_s <= self.end_time_s:
            raise ValueError("time lies outside the flight segment")
        return self.start_position_m + self.velocity_mps * (
            time_s - self.start_time_s
        )


@dataclass(frozen=True)
class ShipborneUavPath:
    ship: ShipMotion
    takeoff_time_s: float
    segments: tuple[LinearFlightSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("an airborne path requires at least one segment")
        first = self.segments[0]
        if not np.isclose(
            first.start_time_s,
            self.takeoff_time_s,
            rtol=0.0,
            atol=TIME_TOLERANCE_S,
        ):
            raise ValueError("first segment must start at takeoff time")
        if not np.allclose(
            first.start_position_m,
            self.ship.position(self.takeoff_time_s),
            rtol=0.0,
            atol=POSITION_TOLERANCE_M,
        ):
            raise ValueError("launch position must equal the ship position at takeoff")
        for previous, following in zip(
            self.segments[:-1],
            self.segments[1:],
            strict=True,
        ):
            if not np.isclose(
                previous.end_time_s,
                following.start_time_s,
                rtol=0.0,
                atol=TIME_TOLERANCE_S,
            ) or not np.allclose(
                previous.end_position_m,
                following.start_position_m,
                rtol=0.0,
                atol=POSITION_TOLERANCE_M,
            ):
                raise ValueError("piecewise flight path must be continuous")

    @property
    def end_time_s(self) -> float:
        return self.segments[-1].end_time_s

    def is_airborne(self, time_s: float) -> bool:
        return self.takeoff_time_s <= time_s <= self.end_time_s

    def segment_at(self, time_s: float) -> LinearFlightSegment:
        for segment in self.segments:
            if segment.start_time_s <= time_s <= segment.end_time_s:
                return segment
        raise ValueError("time lies outside the airborne path")

    def position(self, time_s: float) -> np.ndarray:
        if time_s <= self.takeoff_time_s:
            return self.ship.position(time_s)
        return self.segment_at(time_s).position(time_s)

    def velocity(self, time_s: float) -> np.ndarray:
        if time_s < self.takeoff_time_s:
            direction = np.array(
                [np.cos(self.ship.heading_rad), np.sin(self.ship.heading_rad)]
            )
            return self.ship.speed_mps * direction
        return self.segment_at(time_s).velocity_mps
