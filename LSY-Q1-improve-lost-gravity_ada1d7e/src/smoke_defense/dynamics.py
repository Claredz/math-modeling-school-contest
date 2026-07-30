"""Ship motion and missile pursuit dynamics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np
from scipy.integrate import solve_ivp

from smoke_defense.angles import wrap_to_pi

PositionFunction = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class ShipMotion:
    initial_position_m: tuple[float, float]
    heading_rad: float
    speed_mps: float

    def __post_init__(self) -> None:
        if self.speed_mps < 0:
            raise ValueError("ship speed cannot be negative")

    def position(self, time_s: float) -> np.ndarray:
        direction = np.array([cos(self.heading_rad), sin(self.heading_rad)])
        return np.asarray(self.initial_position_m, dtype=float) + (
            self.speed_mps * float(time_s) * direction
        )


@dataclass(frozen=True)
class MissileGuidanceSpec:
    speed_mps: float
    heading_response_rate_per_s: float
    max_turn_rate_deg_s: float

    def __post_init__(self) -> None:
        if self.speed_mps <= 0:
            raise ValueError("missile speed must be positive")
        if self.heading_response_rate_per_s <= 0:
            raise ValueError("heading response rate must be positive")
        if not 0 < self.max_turn_rate_deg_s <= 180:
            raise ValueError("maximum turn rate must be in (0, 180] deg/s")

    @property
    def max_turn_rate_rad_s(self) -> float:
        return radians(self.max_turn_rate_deg_s)


@dataclass(frozen=True)
class MissileTrajectory:
    start_time_s: float
    end_time_s: float
    times_s: np.ndarray
    states: np.ndarray
    hit_time_s: float | None
    _dense_state: Callable[[float], np.ndarray]

    def state(self, time_s: float) -> np.ndarray:
        if not self.start_time_s <= time_s <= self.end_time_s:
            raise ValueError("requested time lies outside the integrated trajectory")
        return np.asarray(self._dense_state(float(time_s)), dtype=float)

    def position(self, time_s: float) -> np.ndarray:
        return self.state(time_s)[:2]

    def heading(self, time_s: float) -> float:
        state = self.state(time_s)
        if state.size < 3:
            raise ValueError("instantaneous reference has no independent heading state")
        return float(state[2])


def initial_inertial_state(
    missile_position_m: tuple[float, float] | np.ndarray,
    ship_position_m: tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """Set the initial missile heading equal to the initial line of sight."""

    missile = np.asarray(missile_position_m, dtype=float)
    ship = np.asarray(ship_position_m, dtype=float)
    relative = ship - missile
    if np.linalg.norm(relative) == 0:
        raise ValueError("line of sight is undefined at zero relative distance")
    heading = np.arctan2(relative[1], relative[0])
    return np.array([missile[0], missile[1], heading], dtype=float)


def inertial_heading_rate(
    line_of_sight_rad: float,
    heading_rad: float,
    spec: MissileGuidanceSpec,
) -> float:
    error = wrap_to_pi(line_of_sight_rad - heading_rad)
    return float(
        np.clip(
            spec.heading_response_rate_per_s * error,
            -spec.max_turn_rate_rad_s,
            spec.max_turn_rate_rad_s,
        )
    )


def inertial_pursuit_rhs(
    time_s: float,
    state: np.ndarray,
    ship_position: PositionFunction,
    spec: MissileGuidanceSpec,
) -> np.ndarray:
    missile_position = np.asarray(state[:2], dtype=float)
    ship = np.asarray(ship_position(float(time_s)), dtype=float)
    relative = ship - missile_position
    if np.linalg.norm(relative) == 0:
        raise ValueError("line of sight is undefined at zero relative distance")
    line_of_sight = np.arctan2(relative[1], relative[0])
    heading = float(state[2])
    heading_rate = inertial_heading_rate(line_of_sight, heading, spec)
    return np.array(
        [
            spec.speed_mps * np.cos(heading),
            spec.speed_mps * np.sin(heading),
            heading_rate,
        ],
        dtype=float,
    )


def _constant_trajectory(
    state: np.ndarray,
    time_s: float,
) -> MissileTrajectory:
    immutable_state = np.asarray(state, dtype=float)
    return MissileTrajectory(
        start_time_s=time_s,
        end_time_s=time_s,
        times_s=np.array([time_s]),
        states=immutable_state.reshape(1, -1),
        hit_time_s=time_s,
        _dense_state=lambda _time: immutable_state.copy(),
    )


def _integrate(
    *,
    initial_state: np.ndarray,
    appearance_time_s: float,
    ship_position: PositionFunction,
    speed_rhs: Callable[[float, np.ndarray], np.ndarray],
    t_final_s: float,
    hit_radius_m: float,
) -> MissileTrajectory:
    if t_final_s <= appearance_time_s:
        raise ValueError("final time must be after appearance time")
    if hit_radius_m <= 0:
        raise ValueError("hit radius must be positive")
    initial_distance = np.linalg.norm(
        np.asarray(initial_state[:2], dtype=float) - ship_position(appearance_time_s)
    )
    if initial_distance <= hit_radius_m:
        return _constant_trajectory(initial_state, appearance_time_s)

    def hit_event(time_s: float, state: np.ndarray) -> float:
        return float(
            np.linalg.norm(state[:2] - ship_position(time_s)) - hit_radius_m
        )

    hit_event.terminal = True
    hit_event.direction = -1
    solution = solve_ivp(
        speed_rhs,
        (appearance_time_s, t_final_s),
        initial_state,
        events=hit_event,
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.1,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"missile integration failed: {solution.message}")
    hit_time = (
        float(solution.t_events[0][0]) if solution.t_events[0].size else None
    )
    end_time = hit_time if hit_time is not None else float(solution.t[-1])
    return MissileTrajectory(
        start_time_s=appearance_time_s,
        end_time_s=end_time,
        times_s=np.asarray(solution.t, dtype=float),
        states=np.asarray(solution.y.T, dtype=float),
        hit_time_s=hit_time,
        _dense_state=lambda time_s: np.asarray(solution.sol(time_s), dtype=float),
    )


def integrate_inertial_missile(
    initial_position_m: tuple[float, float] | np.ndarray,
    appearance_time_s: float,
    ship_position: PositionFunction,
    spec: MissileGuidanceSpec,
    *,
    t_final_s: float,
    hit_radius_m: float = 80.0,
) -> MissileTrajectory:
    ship_at_appearance = ship_position(appearance_time_s)
    initial_distance = np.linalg.norm(
        np.asarray(initial_position_m, dtype=float) - ship_at_appearance
    )
    if initial_distance <= hit_radius_m:
        relative = ship_at_appearance - np.asarray(initial_position_m, dtype=float)
        heading = 0.0 if np.linalg.norm(relative) == 0 else np.arctan2(
            relative[1],
            relative[0],
        )
        return _constant_trajectory(
            np.array([*np.asarray(initial_position_m, dtype=float), heading]),
            appearance_time_s,
        )
    initial_state = initial_inertial_state(
        initial_position_m,
        ship_at_appearance,
    )
    return _integrate(
        initial_state=initial_state,
        appearance_time_s=appearance_time_s,
        ship_position=ship_position,
        speed_rhs=lambda time_s, state: inertial_pursuit_rhs(
            time_s,
            state,
            ship_position,
            spec,
        ),
        t_final_s=t_final_s,
        hit_radius_m=hit_radius_m,
    )


def integrate_instantaneous_reference(
    initial_position_m: tuple[float, float] | np.ndarray,
    appearance_time_s: float,
    ship_position: PositionFunction,
    *,
    speed_mps: float,
    t_final_s: float,
    hit_radius_m: float = 80.0,
) -> MissileTrajectory:
    if speed_mps <= 0:
        raise ValueError("missile speed must be positive")
    initial_state = np.asarray(initial_position_m, dtype=float)

    def pursuit_rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        relative = np.asarray(ship_position(time_s), dtype=float) - state[:2]
        distance = np.linalg.norm(relative)
        if distance == 0:
            raise ValueError("line of sight is undefined at zero relative distance")
        return speed_mps * relative / distance

    return _integrate(
        initial_state=initial_state,
        appearance_time_s=appearance_time_s,
        ship_position=ship_position,
        speed_rhs=pursuit_rhs,
        t_final_s=t_final_s,
        hit_radius_m=hit_radius_m,
    )
