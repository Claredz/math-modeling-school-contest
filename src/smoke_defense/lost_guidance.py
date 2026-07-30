"""Coupled tracking, smoke-loss, turn-inertia, and reacquisition dynamics.

This module is the Q1 improved model described in
``docs/modeling/q1_lost_coupled_model.md``.  Unlike the conservative baseline,
the missile is not allowed to read the live ship position while it is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import atan2, cos, radians, sin

import numpy as np

from smoke_defense.angles import wrap_to_pi
from smoke_defense.dynamics import ShipMotion
from smoke_defense.smoke import SmokeCloud


class GuidanceMode(StrEnum):
    PRELOCK = "prelock"
    TRACKED = "tracked"
    LOST = "lost"


@dataclass(frozen=True)
class LostGuidanceSpec:
    speed_mps: float = 320.0
    heading_response_rate_per_s: float = 1.0
    max_turn_rate_deg_s: float = 10.0
    tracked_turn_time_constant_s: float = 0.25
    lost_turn_decay_time_s: float = 2.0
    reacquisition_confirm_s: float = 0.5
    detection_range_m: float = 8000.0
    field_of_view_half_angle_deg: float = 15.0
    hit_radius_m: float = 80.0

    def __post_init__(self) -> None:
        positive = (
            self.speed_mps,
            self.heading_response_rate_per_s,
            self.max_turn_rate_deg_s,
            self.tracked_turn_time_constant_s,
            self.lost_turn_decay_time_s,
            self.detection_range_m,
            self.field_of_view_half_angle_deg,
            self.hit_radius_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("lost-guidance physical parameters must be positive")
        if self.reacquisition_confirm_s < 0.0:
            raise ValueError("reacquisition confirmation cannot be negative")
        if self.max_turn_rate_deg_s > 180.0:
            raise ValueError("maximum turn rate cannot exceed 180 deg/s")
        if self.field_of_view_half_angle_deg > 180.0:
            raise ValueError("field-of-view half angle cannot exceed 180 deg")

    @property
    def max_turn_rate_rad_s(self) -> float:
        return radians(self.max_turn_rate_deg_s)

    @property
    def field_of_view_half_angle_rad(self) -> float:
        return radians(self.field_of_view_half_angle_deg)


@dataclass(frozen=True)
class GuidanceEvent:
    time_s: float
    kind: str


@dataclass(frozen=True)
class CoupledTrajectory:
    times_s: np.ndarray
    states: np.ndarray
    modes: tuple[GuidanceMode, ...]
    events: tuple[GuidanceEvent, ...]
    hit_time_s: float | None
    minimum_separation_m: float
    escaped_without_reacquisition: bool

    @property
    def successful_defense(self) -> bool:
        return self.hit_time_s is None and self.escaped_without_reacquisition


def ship_fully_occluded(
    ship_position_m: np.ndarray,
    smoke: SmokeCloud | None,
    time_s: float,
    *,
    ship_radius_m: float,
) -> bool:
    if smoke is None or not smoke.burst_time_s <= time_s <= smoke.failure_time_s:
        return False
    separation = float(np.linalg.norm(ship_position_m - smoke.burst_center_m))
    return separation + ship_radius_m <= smoke.radius(time_s) + 1e-9


def _geometry(
    state: np.ndarray,
    ship_position_m: np.ndarray,
    spec: LostGuidanceSpec,
) -> tuple[float, float, float, bool]:
    relative = ship_position_m - state[:2]
    distance = float(np.linalg.norm(relative))
    line_of_sight = atan2(relative[1], relative[0])
    error = float(wrap_to_pi(line_of_sight - float(state[2])))
    detectable = (
        distance <= spec.detection_range_m
        and abs(error) <= spec.field_of_view_half_angle_rad
    )
    return distance, line_of_sight, error, detectable


def _rhs(
    state: np.ndarray,
    *,
    mode: GuidanceMode,
    line_of_sight_rad: float,
    spec: LostGuidanceSpec,
) -> np.ndarray:
    heading = float(state[2])
    turn_rate = float(state[3])
    if mode is GuidanceMode.TRACKED:
        heading_error = float(wrap_to_pi(line_of_sight_rad - heading))
        commanded_rate = float(
            np.clip(
                spec.heading_response_rate_per_s * heading_error,
                -spec.max_turn_rate_rad_s,
                spec.max_turn_rate_rad_s,
            )
        )
        turn_acceleration = (
            commanded_rate - turn_rate
        ) / spec.tracked_turn_time_constant_s
    else:
        # PRELOCK/LOST states have no live target information.  The last turn
        # rate is retained and exponentially decays instead of snapping to zero.
        turn_acceleration = -turn_rate / spec.lost_turn_decay_time_s
    return np.array(
        [
            spec.speed_mps * cos(heading),
            spec.speed_mps * sin(heading),
            turn_rate,
            turn_acceleration,
        ],
        dtype=float,
    )


def _rk4_step(
    state: np.ndarray,
    dt_s: float,
    *,
    mode: GuidanceMode,
    line_of_sight_rad: float,
    spec: LostGuidanceSpec,
) -> np.ndarray:
    def rhs(value: np.ndarray) -> np.ndarray:
        return _rhs(
            value,
            mode=mode,
            line_of_sight_rad=line_of_sight_rad,
            spec=spec,
        )
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt_s * k1)
    k3 = rhs(state + 0.5 * dt_s * k2)
    k4 = rhs(state + dt_s * k3)
    result = state + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    result[2] = wrap_to_pi(float(result[2]))
    return result


def simulate_lost_coupled_missile(
    *,
    initial_position_m: tuple[float, float] | np.ndarray,
    appearance_time_s: float,
    ship: ShipMotion,
    smoke: SmokeCloud | None,
    spec: LostGuidanceSpec,
    final_time_s: float,
    time_step_s: float = 0.02,
    initial_heading_rad: float | None = None,
) -> CoupledTrajectory:
    """Simulate the endogenous detection window and smoke-induced loss state."""

    if final_time_s <= appearance_time_s:
        raise ValueError("final time must follow appearance time")
    if time_step_s <= 0.0:
        raise ValueError("time step must be positive")
    initial_position = np.asarray(initial_position_m, dtype=float)
    if initial_position.shape != (2,):
        raise ValueError("initial missile position must be two-dimensional")
    initial_relative = ship.position(appearance_time_s) - initial_position
    if np.linalg.norm(initial_relative) == 0.0:
        raise ValueError("initial line of sight is undefined")
    heading = (
        atan2(initial_relative[1], initial_relative[0])
        if initial_heading_rad is None
        else float(initial_heading_rad)
    )
    state = np.array([initial_position[0], initial_position[1], heading, 0.0])
    time_s = float(appearance_time_s)
    distance, _los, _error, detectable = _geometry(
        state, ship.position(time_s), spec
    )
    occluded = ship_fully_occluded(
        ship.position(time_s), smoke, time_s, ship_radius_m=spec.hit_radius_m
    )
    mode = GuidanceMode.TRACKED if detectable and not occluded else GuidanceMode.PRELOCK
    reacquisition_clock_s = 0.0
    has_been_lost = False
    events = [GuidanceEvent(time_s, f"initial_{mode.value}")]
    times = [time_s]
    states = [state.copy()]
    modes = [mode]
    minimum_separation = distance
    hit_time: float | None = None

    while time_s < final_time_s - 1e-12:
        dt_s = min(time_step_s, final_time_s - time_s)
        ship_position = ship.position(time_s)
        distance, line_of_sight, _error, detectable = _geometry(
            state, ship_position, spec
        )
        occluded = ship_fully_occluded(
            ship_position, smoke, time_s, ship_radius_m=spec.hit_radius_m
        )

        if mode is GuidanceMode.TRACKED and (occluded or not detectable):
            mode = GuidanceMode.LOST
            has_been_lost = True
            reacquisition_clock_s = 0.0
            events.append(
                GuidanceEvent(time_s, "smoke_loss" if occluded else "geometry_loss")
            )
        elif mode is not GuidanceMode.TRACKED:
            visible = detectable and not occluded
            reacquisition_clock_s = (
                reacquisition_clock_s + dt_s if visible else 0.0
            )
            if visible and reacquisition_clock_s + 1e-12 >= spec.reacquisition_confirm_s:
                mode = GuidanceMode.TRACKED
                reacquisition_clock_s = 0.0
                events.append(GuidanceEvent(time_s, "reacquisition"))

        state = _rk4_step(
            state,
            dt_s,
            mode=mode,
            line_of_sight_rad=line_of_sight,
            spec=spec,
        )
        time_s += dt_s
        distance = float(np.linalg.norm(state[:2] - ship.position(time_s)))
        minimum_separation = min(minimum_separation, distance)
        times.append(time_s)
        states.append(state.copy())
        modes.append(mode)
        if distance <= spec.hit_radius_m:
            hit_time = time_s
            events.append(GuidanceEvent(time_s, "hit"))
            break

    final_relative = state[:2] - ship.position(time_s)
    missile_velocity = spec.speed_mps * np.array([cos(state[2]), sin(state[2])])
    ship_velocity = ship.speed_mps * np.array(
        [cos(ship.heading_rad), sin(ship.heading_rad)]
    )
    radial_rate = float(
        final_relative @ (missile_velocity - ship_velocity)
        / max(np.linalg.norm(final_relative), 1e-12)
    )
    escaped = (
        has_been_lost
        and hit_time is None
        and mode is not GuidanceMode.TRACKED
        and np.linalg.norm(final_relative) > spec.detection_range_m
        and radial_rate > 0.0
    )
    if escaped:
        events.append(GuidanceEvent(time_s, "escaped_without_reacquisition"))
    return CoupledTrajectory(
        times_s=np.asarray(times),
        states=np.asarray(states),
        modes=tuple(modes),
        events=tuple(events),
        hit_time_s=hit_time,
        minimum_separation_m=minimum_separation,
        escaped_without_reacquisition=escaped,
    )
