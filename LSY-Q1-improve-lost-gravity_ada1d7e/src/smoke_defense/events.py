"""Continuous event roots and closed time intervals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import ceil

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ScalarFunction = Callable[[float], float]


class EventKind(StrEnum):
    APPEARANCE = "appearance"
    DISTANCE_ENTRY = "distance_entry"
    DISTANCE_EXIT = "distance_exit"
    FOV_ENTRY = "fov_entry"
    FOV_EXIT = "fov_exit"
    DETECTION_ENTRY = "detection_entry"
    DETECTION_EXIT = "detection_exit"
    HIT = "hit"
    COMMAND = "command"
    RELEASE = "release"
    BURST = "burst"
    SMOKE_HOLD_END = "smoke_hold_end"
    SMOKE_FAILURE = "smoke_failure"


@dataclass(frozen=True, order=True)
class TrajectoryEvent:
    time_s: float
    kind: EventKind


@dataclass(frozen=True, order=True)
class ClosedInterval:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_s) or not np.isfinite(self.end_s):
            raise ValueError("interval endpoints must be finite")
        if self.end_s < self.start_s:
            raise ValueError("interval end must not precede its start")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, time_s: float) -> bool:
        return self.start_s <= time_s <= self.end_s


def _scan_grid(start_s: float, end_s: float, max_step_s: float) -> np.ndarray:
    if end_s < start_s:
        raise ValueError("scan end must not precede start")
    if max_step_s <= 0:
        raise ValueError("scan step must be positive")
    if end_s == start_s:
        return np.array([start_s], dtype=float)
    count = max(1, ceil((end_s - start_s) / max_step_s))
    return np.linspace(start_s, end_s, count + 1)


def find_roots(
    function: ScalarFunction,
    *,
    start_s: float,
    end_s: float,
    max_step_s: float,
    lipschitz_bound_per_s: float,
    value_tolerance: float = 1e-12,
    time_tolerance_s: float = 1e-7,
) -> tuple[float, ...]:
    """Find all roots resolved above ``time_tolerance_s``.

    A cell is discarded only when the declared global Lipschitz bound proves
    that its midpoint enclosure cannot contain zero. Every other cell is
    bisected adaptively, so a short sign excursion with same-sign coarse
    endpoints is retained rather than skipped.
    """

    if lipschitz_bound_per_s < 0:
        raise ValueError("Lipschitz bound cannot be negative")
    if value_tolerance <= 0:
        raise ValueError("value tolerance must be positive")
    if time_tolerance_s <= 0:
        raise ValueError("time tolerance must be positive")
    grid = _scan_grid(start_s, end_s, max_step_s)
    value_cache: dict[float, float] = {}

    def value(time_s: float) -> float:
        key = float(time_s)
        if key not in value_cache:
            value_cache[key] = float(function(key))
        return value_cache[key]

    roots: list[float] = []

    def add_leaf_roots(
        left_s: float,
        left_value: float,
        midpoint_s: float,
        midpoint_value: float,
        right_s: float,
        right_value: float,
    ) -> None:
        samples = (
            (left_s, left_value),
            (midpoint_s, midpoint_value),
            (right_s, right_value),
        )
        for time_s, sample_value in samples:
            if abs(sample_value) <= value_tolerance:
                roots.append(time_s)
        for (first_s, first_value), (second_s, second_value) in zip(
            samples[:-1],
            samples[1:],
            strict=True,
        ):
            if first_value * second_value < 0:
                roots.append(
                    float(
                        brentq(
                            function,
                            first_s,
                            second_s,
                            xtol=1e-13,
                            rtol=1e-14,
                        )
                    )
                )
        if all(
            first_value * second_value >= 0
            for (_, first_value), (_, second_value) in zip(
                samples[:-1],
                samples[1:],
                strict=True,
            )
        ):
            minimum = minimize_scalar(
                lambda time_s: abs(float(function(float(time_s)))),
                bounds=(left_s, right_s),
                method="bounded",
                options={"xatol": time_tolerance_s / 8.0},
            )
            if minimum.success and minimum.fun <= value_tolerance:
                roots.append(float(minimum.x))

    def search_cell(
        left_s: float,
        left_value: float,
        right_s: float,
        right_value: float,
    ) -> None:
        midpoint_s = 0.5 * (left_s + right_s)
        midpoint_value = value(midpoint_s)
        half_width_s = 0.5 * (right_s - left_s)
        if (
            abs(midpoint_value)
            > lipschitz_bound_per_s * half_width_s + value_tolerance
        ):
            return
        if max(
            abs(left_value),
            abs(midpoint_value),
            abs(right_value),
        ) <= value_tolerance:
            roots.extend((left_s, right_s))
            return
        if right_s - left_s <= time_tolerance_s:
            add_leaf_roots(
                left_s,
                left_value,
                midpoint_s,
                midpoint_value,
                right_s,
                right_value,
            )
            return
        search_cell(
            left_s,
            left_value,
            midpoint_s,
            midpoint_value,
        )
        search_cell(
            midpoint_s,
            midpoint_value,
            right_s,
            right_value,
        )

    for left_s, right_s in zip(grid[:-1], grid[1:], strict=True):
        search_cell(
            float(left_s),
            value(float(left_s)),
            float(right_s),
            value(float(right_s)),
        )
    if grid.size == 1 and abs(value(float(grid[0]))) <= value_tolerance:
        roots.append(float(grid[0]))

    unique: list[float] = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > 1e-9:
            unique.append(root)
    return tuple(unique)


def _crossing_kind(
    function: ScalarFunction,
    root_s: float,
    start_s: float,
    end_s: float,
    entry_kind: EventKind,
    exit_kind: EventKind,
) -> EventKind | None:
    span = max(end_s - start_s, 1.0)
    epsilon = min(1e-6 * span, 1e-5)
    left_s = max(start_s, root_s - epsilon)
    right_s = min(end_s, root_s + epsilon)
    left_value = float(function(left_s))
    right_value = float(function(right_s))
    if left_value < 0 <= right_value:
        return entry_kind
    if left_value >= 0 > right_value:
        return exit_kind
    return None


def find_boundary_events(
    function: ScalarFunction,
    *,
    start_s: float,
    end_s: float,
    max_step_s: float,
    lipschitz_bound_per_s: float,
    entry_kind: EventKind,
    exit_kind: EventKind,
) -> tuple[TrajectoryEvent, ...]:
    events = []
    for root_s in find_roots(
        function,
        start_s=start_s,
        end_s=end_s,
        max_step_s=max_step_s,
        lipschitz_bound_per_s=lipschitz_bound_per_s,
    ):
        kind = _crossing_kind(
            function,
            root_s,
            start_s,
            end_s,
            entry_kind,
            exit_kind,
        )
        if kind is not None:
            events.append(TrajectoryEvent(root_s, kind))
    return tuple(events)


def intervals_where_nonnegative(
    function: ScalarFunction,
    *,
    start_s: float,
    end_s: float,
    max_step_s: float,
    lipschitz_bound_per_s: float,
    value_tolerance: float = 1e-10,
) -> tuple[ClosedInterval, ...]:
    """Return the closed components on which a continuous margin is nonnegative."""

    if end_s == start_s:
        if function(start_s) >= -value_tolerance:
            return (ClosedInterval(start_s, end_s),)
        return ()

    roots = find_roots(
        function,
        start_s=start_s,
        end_s=end_s,
        max_step_s=max_step_s,
        lipschitz_bound_per_s=lipschitz_bound_per_s,
        value_tolerance=value_tolerance,
    )
    breakpoints = [start_s, *roots, end_s]
    breakpoints = sorted({round(float(time_s), 12) for time_s in breakpoints})
    intervals: list[ClosedInterval] = []
    for left_s, right_s in zip(
        breakpoints[:-1],
        breakpoints[1:],
        strict=True,
    ):
        midpoint_s = 0.5 * (left_s + right_s)
        if float(function(midpoint_s)) >= -value_tolerance:
            candidate = ClosedInterval(left_s, right_s)
            if intervals and abs(intervals[-1].end_s - candidate.start_s) <= 1e-9:
                intervals[-1] = ClosedInterval(
                    intervals[-1].start_s,
                    candidate.end_s,
                )
            else:
                intervals.append(candidate)
    return tuple(intervals)
