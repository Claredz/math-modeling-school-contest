"""Continuous-time coverage verification over closed detection components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from smoke_defense.coverage import (
    CertificationStatus,
    Disk,
    UnionGapEvaluation,
    evaluate_union_gap_at_time,
    single_smoke_gap,
)
from smoke_defense.events import ClosedInterval, TrajectoryEvent
from smoke_defense.smoke import SmokeCloud

PositionFunction = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class ContinuousCoverageCertificate:
    status: CertificationStatus
    witness_time_s: float | None = None
    maximum_sampled_gap_m: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class CoverageModeCertificate:
    interval: ClosedInterval
    active_smoke_indices: tuple[int, ...]
    status: CertificationStatus
    maximum_gap_lower_bound_m: float
    maximum_gap_upper_bound_m: float
    witness_time_s: float | None = None
    witness_m: np.ndarray | None = None
    reason: str = ""


@dataclass(frozen=True)
class MultiSmokeCoverageCertificate:
    status: CertificationStatus
    checked_event_times_s: tuple[float, ...]
    mode_certificates: tuple[CoverageModeCertificate, ...]
    maximum_gap_lower_bound_m: float
    maximum_gap_upper_bound_m: float
    minimum_margin_m: float
    exposed_intervals: tuple[ClosedInterval, ...]
    unresolved_intervals: tuple[ClosedInterval, ...]
    total_exposed_duration_s: float
    maximum_exposed_interval_s: float
    maximum_exposed_interval: ClosedInterval | None = None
    witness_time_s: float | None = None
    witness_m: np.ndarray | None = None
    reason: str = ""


def _critical_times(
    component: ClosedInterval,
    smoke: SmokeCloud,
) -> tuple[float, ...]:
    candidates = {
        component.start_s,
        component.end_s,
        smoke.burst_time_s,
        smoke.hold_end_time_s,
        smoke.failure_time_s,
    }
    return tuple(
        sorted(
            time_s
            for time_s in candidates
            if component.start_s <= time_s <= component.end_s
        )
    )


def certify_single_smoke_continuous_coverage(
    *,
    ship_position: PositionFunction,
    smoke: SmokeCloud,
    detection_components: tuple[ClosedInterval, ...],
    ship_radius_m: float = 80.0,
    ship_speed_bound_mps: float = 7.71,
    time_tolerance_s: float = 1e-3,
) -> ContinuousCoverageCertificate:
    """Use point witnesses for failure and a Lipschitz bound for full coverage."""

    if ship_speed_bound_mps < 0:
        raise ValueError("ship speed bound cannot be negative")
    if time_tolerance_s <= 0:
        raise ValueError("time tolerance must be positive")
    if not detection_components:
        return ContinuousCoverageCertificate(
            CertificationStatus.CERTIFIED_FEASIBLE,
            reason="empty detection set is vacuously covered",
        )

    maximum_sampled_gap = -np.inf

    def gap(time_s: float) -> float:
        return single_smoke_gap(
            ship_position(time_s),
            smoke.burst_center_m,
            smoke.radius(time_s),
            ship_radius_m=ship_radius_m,
        )

    unresolved = False
    for component in detection_components:
        critical = _critical_times(component, smoke)
        if len(critical) == 1:
            critical = (component.start_s, component.end_s)
        for time_s in critical:
            value = gap(time_s)
            maximum_sampled_gap = max(maximum_sampled_gap, value)
            if value > 0:
                return ContinuousCoverageCertificate(
                    CertificationStatus.CERTIFIED_INFEASIBLE,
                    witness_time_s=time_s,
                    maximum_sampled_gap_m=maximum_sampled_gap,
                    reason="closed detection endpoint has a positive coverage gap",
                )

        stack = list(zip(critical[:-1], critical[1:], strict=True))
        while stack:
            left_s, right_s = stack.pop()
            if right_s == left_s:
                continue
            midpoint_s = 0.5 * (left_s + right_s)
            values = (gap(left_s), gap(midpoint_s), gap(right_s))
            maximum_sampled_gap = max(maximum_sampled_gap, *values)
            positive_index = next(
                (index for index, value in enumerate(values) if value > 0),
                None,
            )
            if positive_index is not None:
                witness_times = (left_s, midpoint_s, right_s)
                return ContinuousCoverageCertificate(
                    CertificationStatus.CERTIFIED_INFEASIBLE,
                    witness_time_s=witness_times[positive_index],
                    maximum_sampled_gap_m=maximum_sampled_gap,
                    reason="continuous interval contains an exact positive-gap witness",
                )

            in_decay = (
                smoke.hold_end_time_s < midpoint_s < smoke.failure_time_s
            )
            radius_rate_bound = smoke.decay_rate_mps if in_decay else 0.0
            lipschitz_bound = ship_speed_bound_mps + radius_rate_bound
            upper_bound = max(values) + lipschitz_bound * (right_s - left_s) / 4.0
            if upper_bound <= 0:
                continue
            if right_s - left_s <= time_tolerance_s:
                unresolved = True
                continue
            stack.append((left_s, midpoint_s))
            stack.append((midpoint_s, right_s))

    if unresolved:
        return ContinuousCoverageCertificate(
            CertificationStatus.INDETERMINATE,
            maximum_sampled_gap_m=maximum_sampled_gap,
            reason="Lipschitz enclosure straddles zero at time tolerance",
        )
    return ContinuousCoverageCertificate(
        CertificationStatus.CERTIFIED_FEASIBLE,
        maximum_sampled_gap_m=maximum_sampled_gap,
        reason="all event-split intervals have nonpositive Lipschitz upper bounds",
    )


def _merge_intervals(
    intervals: list[ClosedInterval],
    *,
    tolerance_s: float = 1e-12,
) -> tuple[ClosedInterval, ...]:
    merged: list[ClosedInterval] = []
    for interval in sorted(intervals):
        if (
            merged
            and interval.start_s <= merged[-1].end_s + tolerance_s
        ):
            merged[-1] = ClosedInterval(
                merged[-1].start_s,
                max(merged[-1].end_s, interval.end_s),
            )
        else:
            merged.append(interval)
    return tuple(merged)


def certify_multi_smoke_coverage(
    *,
    ship_position: PositionFunction,
    smokes: tuple[SmokeCloud, ...],
    detection_components: tuple[ClosedInterval, ...],
    source_events: tuple[TrajectoryEvent, ...] = (),
    ship_radius_m: float = 80.0,
    ship_speed_bound_mps: float = 7.71,
    spatial_tolerance_m: float = 0.05,
    time_tolerance_s: float = 1e-3,
    initial_polygon_sides: int = 32,
    maximum_polygon_sides: int = 2048,
) -> MultiSmokeCoverageCertificate:
    """Certify multi-smoke coverage over closed, event-split detection sets."""

    if ship_radius_m < 0:
        raise ValueError("ship radius cannot be negative")
    if ship_speed_bound_mps < 0:
        raise ValueError("ship speed bound cannot be negative")
    if time_tolerance_s <= 0:
        raise ValueError("time tolerance must be positive")
    if not detection_components:
        return MultiSmokeCoverageCertificate(
            status=CertificationStatus.CERTIFIED_FEASIBLE,
            checked_event_times_s=(),
            mode_certificates=(),
            maximum_gap_lower_bound_m=float("-inf"),
            maximum_gap_upper_bound_m=float("-inf"),
            minimum_margin_m=float("inf"),
            exposed_intervals=(),
            unresolved_intervals=(),
            total_exposed_duration_s=0.0,
            maximum_exposed_interval_s=0.0,
            reason="empty detection set is vacuously covered",
        )

    cache: dict[float, UnionGapEvaluation] = {}

    def evaluate(time_s: float) -> UnionGapEvaluation:
        if time_s not in cache:
            disks = tuple(
                Disk(smoke.burst_center_m, radius_m)
                for smoke in smokes
                if (radius_m := smoke.radius(time_s)) > 0.0
            )
            cache[time_s] = evaluate_union_gap_at_time(
                Disk(ship_position(time_s), ship_radius_m),
                disks,
                spatial_tolerance_m=spatial_tolerance_m,
                initial_polygon_sides=initial_polygon_sides,
                maximum_polygon_sides=maximum_polygon_sides,
            )
        return cache[time_s]

    split_times: set[float] = set()
    for component in detection_components:
        split_times.update((component.start_s, component.end_s))
        split_times.update(
            event.time_s
            for event in source_events
            if component.contains(event.time_s)
        )
        for smoke in smokes:
            split_times.update(
                time_s
                for time_s in (
                    smoke.burst_time_s,
                    smoke.hold_end_time_s,
                    smoke.failure_time_s,
                )
                if component.contains(time_s)
            )
    checked_event_times = tuple(sorted(split_times))
    event_evaluations = {
        time_s: evaluate(time_s) for time_s in checked_event_times
    }

    exposed_cells: list[ClosedInterval] = []
    unresolved_cells: list[ClosedInterval] = []
    mode_certificates: list[CoverageModeCertificate] = []
    global_lower_bounds = [
        item.lower_bound_m for item in event_evaluations.values()
    ]
    global_upper_bounds = [
        item.upper_bound_m for item in event_evaluations.values()
    ]
    witness_candidates: list[tuple[float, float, np.ndarray | None]] = [
        (item.lower_bound_m, time_s, item.witness_m)
        for time_s, item in event_evaluations.items()
        if item.status is CertificationStatus.CERTIFIED_INFEASIBLE
    ]

    for component in detection_components:
        component_splits = sorted(
            time_s
            for time_s in split_times
            if component.contains(time_s)
        )
        for left_event_s, right_event_s in zip(
            component_splits[:-1],
            component_splits[1:],
            strict=True,
        ):
            if right_event_s <= left_event_s:
                continue
            left_s = float(np.nextafter(left_event_s, right_event_s))
            right_s = float(np.nextafter(right_event_s, left_event_s))
            if right_s <= left_s:
                continue
            midpoint_s = 0.5 * (left_s + right_s)
            active_indices = tuple(
                index
                for index, smoke in enumerate(smokes)
                if smoke.radius(midpoint_s) > 0.0
            )
            radius_rate_bound = max(
                (
                    smoke.decay_rate_mps
                    for smoke in smokes
                    if (
                        smoke.hold_end_time_s
                        < midpoint_s
                        < smoke.failure_time_s
                    )
                ),
                default=0.0,
            )
            lipschitz_bound = ship_speed_bound_mps + radius_rate_bound
            stack = [(left_s, right_s)]
            mode_lower_bounds: list[float] = []
            mode_upper_bounds: list[float] = []
            mode_unresolved = False
            mode_exposed = False
            mode_witness: tuple[float, np.ndarray | None] | None = None

            while stack:
                cell_left_s, cell_right_s = stack.pop()
                cell_midpoint_s = 0.5 * (cell_left_s + cell_right_s)
                sample_times = (
                    cell_left_s,
                    cell_midpoint_s,
                    cell_right_s,
                )
                samples = tuple(evaluate(time_s) for time_s in sample_times)
                width_s = cell_right_s - cell_left_s
                lower_bound_m = max(
                    sample.lower_bound_m for sample in samples
                )
                upper_bound_m = (
                    max(sample.upper_bound_m for sample in samples)
                    + lipschitz_bound * width_s / 4.0
                )
                exposed_lower_bound_m = (
                    min(sample.lower_bound_m for sample in samples)
                    - lipschitz_bound * width_s / 4.0
                )
                mode_lower_bounds.append(lower_bound_m)
                mode_upper_bounds.append(upper_bound_m)
                positive_index = next(
                    (
                        index
                        for index, sample in enumerate(samples)
                        if sample.status
                        is CertificationStatus.CERTIFIED_INFEASIBLE
                    ),
                    None,
                )
                if positive_index is not None:
                    sample = samples[positive_index]
                    candidate = (
                        sample_times[positive_index],
                        sample.witness_m,
                    )
                    if mode_witness is None:
                        mode_witness = candidate
                    witness_candidates.append(
                        (
                            sample.lower_bound_m,
                            sample_times[positive_index],
                            sample.witness_m,
                        )
                    )
                if upper_bound_m <= 0.0:
                    continue
                if exposed_lower_bound_m > 0.0:
                    mode_exposed = True
                    exposed_cells.append(
                        ClosedInterval(cell_left_s, cell_right_s)
                    )
                    continue
                if width_s <= time_tolerance_s:
                    mode_unresolved = True
                    unresolved_cells.append(
                        ClosedInterval(cell_left_s, cell_right_s)
                    )
                    continue
                stack.append((cell_left_s, cell_midpoint_s))
                stack.append((cell_midpoint_s, cell_right_s))

            if mode_exposed or mode_witness is not None:
                mode_status = CertificationStatus.CERTIFIED_INFEASIBLE
            elif mode_unresolved:
                mode_status = CertificationStatus.INDETERMINATE
            else:
                mode_status = CertificationStatus.CERTIFIED_FEASIBLE
            mode_interval = ClosedInterval(left_s, right_s)
            mode_lower = max(mode_lower_bounds, default=float("-inf"))
            mode_upper = max(mode_upper_bounds, default=float("-inf"))
            global_lower_bounds.append(mode_lower)
            global_upper_bounds.append(mode_upper)
            mode_certificates.append(
                CoverageModeCertificate(
                    interval=mode_interval,
                    active_smoke_indices=active_indices,
                    status=mode_status,
                    maximum_gap_lower_bound_m=mode_lower,
                    maximum_gap_upper_bound_m=mode_upper,
                    witness_time_s=(
                        mode_witness[0] if mode_witness is not None else None
                    ),
                    witness_m=(
                        mode_witness[1] if mode_witness is not None else None
                    ),
                    reason="adaptive Lipschitz enclosure within one event mode",
                )
            )

    event_infeasible = any(
        item.status is CertificationStatus.CERTIFIED_INFEASIBLE
        for item in event_evaluations.values()
    )
    event_indeterminate = any(
        item.status is CertificationStatus.INDETERMINATE
        and item.upper_bound_m > 0.0
        for item in event_evaluations.values()
    )
    mode_infeasible = any(
        item.status is CertificationStatus.CERTIFIED_INFEASIBLE
        for item in mode_certificates
    )
    mode_indeterminate = any(
        item.status is CertificationStatus.INDETERMINATE
        for item in mode_certificates
    )
    if event_infeasible or mode_infeasible:
        status = CertificationStatus.CERTIFIED_INFEASIBLE
    elif event_indeterminate or mode_indeterminate:
        status = CertificationStatus.INDETERMINATE
    else:
        status = CertificationStatus.CERTIFIED_FEASIBLE

    exposed_intervals = _merge_intervals(exposed_cells)
    unresolved_intervals = _merge_intervals(unresolved_cells)
    maximum_exposed_interval = max(
        exposed_intervals,
        key=lambda interval: interval.duration_s,
        default=None,
    )
    maximum_gap_lower = max(global_lower_bounds, default=float("-inf"))
    maximum_gap_upper = max(global_upper_bounds, default=float("-inf"))
    witness_time_s: float | None = None
    witness_m: np.ndarray | None = None
    if witness_candidates:
        _gap, witness_time_s, witness_m = max(
            witness_candidates,
            key=lambda item: item[0],
        )
    return MultiSmokeCoverageCertificate(
        status=status,
        checked_event_times_s=checked_event_times,
        mode_certificates=tuple(mode_certificates),
        maximum_gap_lower_bound_m=maximum_gap_lower,
        maximum_gap_upper_bound_m=maximum_gap_upper,
        minimum_margin_m=-maximum_gap_upper,
        exposed_intervals=exposed_intervals,
        unresolved_intervals=unresolved_intervals,
        total_exposed_duration_s=sum(
            interval.duration_s for interval in exposed_intervals
        ),
        maximum_exposed_interval_s=(
            maximum_exposed_interval.duration_s
            if maximum_exposed_interval is not None
            else 0.0
        ),
        maximum_exposed_interval=maximum_exposed_interval,
        witness_time_s=witness_time_s,
        witness_m=witness_m,
        reason="closed events and continuous modes were certified separately",
    )
