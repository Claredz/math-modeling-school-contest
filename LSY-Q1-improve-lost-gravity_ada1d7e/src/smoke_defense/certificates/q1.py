"""Analytic single-smoke duration and causal-availability certificates."""

from __future__ import annotations

from dataclasses import dataclass

from smoke_defense.coverage import CertificationStatus
from smoke_defense.events import ClosedInterval


@dataclass(frozen=True)
class SingleSmokeDurationCertificate:
    status: CertificationStatus
    limit_s: float
    longest_component_s: float
    reason: str


@dataclass(frozen=True)
class EarliestSmokeCertificate:
    status: CertificationStatus
    earliest_burst_time_s: float
    unavoidable_exposure_s: float
    reason: str


def certify_single_smoke_duration(
    components: tuple[ClosedInterval, ...],
    *,
    ship_speed_mps: float = 7.71,
    maximum_smoke_radius_m: float = 120.0,
    ship_radius_m: float = 80.0,
    tolerance_s: float = 1e-9,
) -> SingleSmokeDurationCertificate:
    """Apply the fixed-smoke chord-length upper bound component by component."""

    if ship_speed_mps <= 0:
        raise ValueError("ship speed must be positive")
    cover_half_width_m = maximum_smoke_radius_m - ship_radius_m
    if cover_half_width_m < 0:
        limit_s = 0.0
    else:
        limit_s = 2.0 * cover_half_width_m / ship_speed_mps
    longest = max((component.duration_s for component in components), default=0.0)
    if longest > limit_s + tolerance_s:
        return SingleSmokeDurationCertificate(
            CertificationStatus.CERTIFIED_INFEASIBLE,
            limit_s,
            longest,
            "detection_component_exceeds_single_smoke_bound",
        )
    return SingleSmokeDurationCertificate(
        CertificationStatus.INDETERMINATE,
        limit_s,
        longest,
        "duration_bound_does_not_prove_feasibility",
    )


def _intersection_duration(
    components: tuple[ClosedInterval, ...],
    start_s: float,
    end_s: float,
) -> float:
    return sum(
        max(
            0.0,
            min(component.end_s, end_s) - max(component.start_s, start_s),
        )
        for component in components
    )


def certify_earliest_smoke_availability(
    components: tuple[ClosedInterval, ...],
    *,
    command_time_s: float,
    minimum_release_response_s: float = 2.0,
    detonation_delay_s: float = 3.5,
    no_predeployed_smoke: bool = True,
    full_detection_window_required: bool = True,
    tolerance_s: float = 1e-9,
) -> EarliestSmokeCertificate:
    earliest_burst = (
        command_time_s
        + minimum_release_response_s
        + detonation_delay_s
    )
    unavoidable = _intersection_duration(
        components,
        command_time_s,
        earliest_burst,
    )
    has_preburst_detection = any(
        max(component.start_s, command_time_s)
        <= min(component.end_s, earliest_burst)
        and max(component.start_s, command_time_s) < earliest_burst
        for component in components
    )
    if (
        no_predeployed_smoke
        and full_detection_window_required
        and (unavoidable > tolerance_s or has_preburst_detection)
    ):
        return EarliestSmokeCertificate(
            CertificationStatus.CERTIFIED_INFEASIBLE,
            earliest_burst,
            unavoidable,
            "detection_precedes_earliest_possible_smoke",
        )
    return EarliestSmokeCertificate(
        CertificationStatus.INDETERMINATE,
        earliest_burst,
        unavoidable,
        "causal_bound_does_not_prove_feasibility",
    )
