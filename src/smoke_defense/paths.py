"""Continuous piecewise-linear paths for UAVs launched from the moving ship."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

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

    @property
    def flight_distance_m(self) -> float:
        return sum(
            float(
                np.linalg.norm(
                    segment.end_position_m - segment.start_position_m
                )
            )
            for segment in self.segments
        )

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


@dataclass(frozen=True)
class ReleaseWaypoint:
    release_time_s: float
    release_position_m: np.ndarray
    heading_unit: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.release_position_m, dtype=float).copy()
        heading = np.asarray(self.heading_unit, dtype=float).copy()
        if position.shape != (2,) or heading.shape != (2,):
            raise ValueError("release position and heading must be two-dimensional")
        if not np.isclose(
            np.linalg.norm(heading),
            1.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("release heading must be a unit vector")
        object.__setattr__(self, "release_position_m", position)
        object.__setattr__(self, "heading_unit", heading)


def _connect_with_terminal_heading(
    *,
    start_time_s: float,
    start_position_m: np.ndarray,
    release: ReleaseWaypoint,
) -> tuple[LinearFlightSegment, ...]:
    duration_s = release.release_time_s - start_time_s
    if duration_s <= 0.0:
        raise ValueError("release times must be strictly increasing")
    available_length_m = UAV_SPEED_MPS * duration_s
    displacement = release.release_position_m - start_position_m
    direct_distance_m = float(np.linalg.norm(displacement))
    if direct_distance_m > available_length_m + POSITION_TOLERANCE_M:
        raise ValueError("ordered release is unreachable from the previous event")

    direct_heading = (
        displacement / direct_distance_m
        if direct_distance_m > POSITION_TOLERANCE_M
        else release.heading_unit
    )
    if np.isclose(
        direct_distance_m,
        available_length_m,
        rtol=0.0,
        atol=POSITION_TOLERANCE_M,
    ):
        if not np.allclose(
            direct_heading,
            release.heading_unit,
            rtol=0.0,
            atol=POSITION_TOLERANCE_M,
        ):
            raise ValueError(
                "ordered release is unreachable with its required terminal heading"
            )
        return (
            LinearFlightSegment(
                start_time_s,
                release.release_time_s,
                start_position_m,
                release.release_position_m,
            ),
        )

    def length_residual(final_leg_m: float) -> float:
        waypoint = (
            release.release_position_m
            - final_leg_m * release.heading_unit
        )
        return (
            float(np.linalg.norm(waypoint - start_position_m))
            + final_leg_m
            - available_length_m
        )

    final_leg_m = float(
        brentq(
            length_residual,
            0.0,
            available_length_m,
            xtol=1e-12,
        )
    )
    waypoint = release.release_position_m - final_leg_m * release.heading_unit
    first_leg_m = float(np.linalg.norm(waypoint - start_position_m))
    segments: list[LinearFlightSegment] = []
    current_time_s = start_time_s
    if first_leg_m > POSITION_TOLERANCE_M:
        first_end_time_s = current_time_s + first_leg_m / UAV_SPEED_MPS
        segments.append(
            LinearFlightSegment(
                current_time_s,
                first_end_time_s,
                start_position_m,
                waypoint,
            )
        )
        current_time_s = first_end_time_s
    if final_leg_m <= POSITION_TOLERANCE_M:
        raise ValueError(
            "ordered release timing leaves no terminal-heading segment"
        )
    segments.append(
        LinearFlightSegment(
            current_time_s,
            release.release_time_s,
            waypoint,
            release.release_position_m,
        )
    )
    return tuple(segments)


def build_ordered_release_path(
    *,
    ship: ShipMotion,
    takeoff_time_s: float,
    releases: tuple[ReleaseWaypoint, ...],
    continue_until_s: float,
) -> ShipborneUavPath:
    """Connect ordered release events with one fixed-speed shipborne path."""

    if not 1 <= len(releases) <= 3:
        raise ValueError("an ordered path requires one to three releases")
    if continue_until_s <= releases[-1].release_time_s:
        raise ValueError("path must continue after the final release")
    segments: list[LinearFlightSegment] = []
    current_time_s = takeoff_time_s
    current_position = ship.position(takeoff_time_s)
    for release in releases:
        connector = _connect_with_terminal_heading(
            start_time_s=current_time_s,
            start_position_m=current_position,
            release=release,
        )
        segments.extend(connector)
        current_time_s = release.release_time_s
        current_position = release.release_position_m

    final_heading = releases[-1].heading_unit
    post_release_position = current_position + (
        UAV_SPEED_MPS
        * (continue_until_s - current_time_s)
        * final_heading
    )
    segments.append(
        LinearFlightSegment(
            current_time_s,
            continue_until_s,
            current_position,
            post_release_position,
        )
    )
    return ShipborneUavPath(
        ship=ship,
        takeoff_time_s=takeoff_time_s,
        segments=tuple(segments),
    )
