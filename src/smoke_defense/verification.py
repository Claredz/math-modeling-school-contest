"""Continuous-time coverage verification over closed detection components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from smoke_defense.coverage import CertificationStatus, single_smoke_gap
from smoke_defense.events import ClosedInterval
from smoke_defense.smoke import SmokeCloud

PositionFunction = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class ContinuousCoverageCertificate:
    status: CertificationStatus
    witness_time_s: float | None = None
    maximum_sampled_gap_m: float | None = None
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
