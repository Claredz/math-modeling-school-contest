"""Low-dimensional Q1 candidate construction and lexicographic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import numpy as np
from scipy.optimize import brentq

from smoke_defense.coverage import CertificationStatus, single_smoke_gap
from smoke_defense.detection import DetectionSet
from smoke_defense.dynamics import ShipMotion
from smoke_defense.events import ClosedInterval, intervals_where_nonnegative
from smoke_defense.path_constraints import certify_operation_radius
from smoke_defense.paths import (
    UAV_SPEED_MPS,
    LinearFlightSegment,
    ShipborneUavPath,
)
from smoke_defense.smoke import SmokeCloud, detonation_position
from smoke_defense.timeline import BombEvents
from smoke_defense.verification import certify_single_smoke_continuous_coverage


@dataclass(frozen=True)
class Q1Candidate:
    strict_status: CertificationStatus
    covered_duration_s: float
    exposed_duration_s: float
    maximum_exposed_interval_s: float
    minimum_margin_m: float
    flight_distance_m: float
    command_time_s: float = 0.0
    takeoff_time_s: float = 0.0
    release_time_s: float = 0.0
    burst_time_s: float = 0.0
    coverage_center_time_s: float = 0.0
    release_position_m: np.ndarray | None = None
    burst_center_m: np.ndarray | None = None
    burst_heading: np.ndarray | None = None
    covered_intervals: tuple[ClosedInterval, ...] = ()
    path: ShipborneUavPath | None = None
    smoke: SmokeCloud | None = None
    reachability_status: str = "not_evaluated"

    @classmethod
    def for_ranking_test(
        cls,
        *,
        strict_status: CertificationStatus,
        covered_duration_s: float,
        maximum_exposed_interval_s: float,
        minimum_margin_m: float,
        flight_distance_m: float,
    ) -> Q1Candidate:
        return cls(
            strict_status=strict_status,
            covered_duration_s=covered_duration_s,
            exposed_duration_s=0.0,
            maximum_exposed_interval_s=maximum_exposed_interval_s,
            minimum_margin_m=minimum_margin_m,
            flight_distance_m=flight_distance_m,
        )


def candidate_rank_key(candidate: Q1Candidate) -> tuple[float, ...]:
    strict_rank = {
        CertificationStatus.CERTIFIED_FEASIBLE: 2.0,
        CertificationStatus.INDETERMINATE: 1.0,
        CertificationStatus.CERTIFIED_INFEASIBLE: 0.0,
    }[candidate.strict_status]
    return (
        strict_rank,
        candidate.covered_duration_s,
        -candidate.maximum_exposed_interval_s,
        candidate.minimum_margin_m,
        -candidate.flight_distance_m,
    )


def _unit_direction(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return np.array([1.0, 0.0])
    return vector / norm


def build_shipborne_release_path(
    *,
    ship: ShipMotion,
    takeoff_time_s: float,
    release_time_s: float,
    burst_center_m: np.ndarray,
    detonation_delay_s: float,
) -> tuple[ShipborneUavPath, np.ndarray, np.ndarray]:
    """Construct a one/two-segment fixed-speed path with the required exit heading."""

    if release_time_s <= takeoff_time_s:
        raise ValueError("release must occur after takeoff")
    start = ship.position(takeoff_time_s)
    burst_center = np.asarray(burst_center_m, dtype=float)
    burst_heading = _unit_direction(burst_center - start)
    release_position = (
        burst_center
        - UAV_SPEED_MPS * detonation_delay_s * burst_heading
    )
    available_length = UAV_SPEED_MPS * (release_time_s - takeoff_time_s)
    direct_distance = float(np.linalg.norm(release_position - start))
    if direct_distance > available_length + 1e-8:
        raise ValueError("release point is unreachable by the shipborne UAV")

    direct_heading = _unit_direction(release_position - start)
    if isclose(direct_distance, available_length, abs_tol=1e-8) and np.allclose(
        direct_heading,
        burst_heading,
        rtol=0.0,
        atol=1e-8,
    ):
        segments = (
            LinearFlightSegment(
                takeoff_time_s,
                release_time_s,
                start,
                release_position,
            ),
        )
    else:
        relative_release = release_position - start

        def path_length_residual(final_leg_m: float) -> float:
            waypoint_relative = (
                relative_release - final_leg_m * burst_heading
            )
            return (
                float(np.linalg.norm(waypoint_relative))
                + final_leg_m
                - available_length
            )

        final_leg_m = float(
            brentq(
                path_length_residual,
                0.0,
                available_length,
                xtol=1e-12,
            )
        )
        waypoint = release_position - final_leg_m * burst_heading
        first_leg_m = float(np.linalg.norm(waypoint - start))
        if first_leg_m <= 1e-8 or final_leg_m <= 1e-8:
            raise ValueError("release timing leaves no positive final heading segment")
        turn_time_s = takeoff_time_s + first_leg_m / UAV_SPEED_MPS
        segments = (
            LinearFlightSegment(
                takeoff_time_s,
                turn_time_s,
                start,
                waypoint,
            ),
            LinearFlightSegment(
                turn_time_s,
                release_time_s,
                waypoint,
                release_position,
            ),
        )

    path = ShipborneUavPath(
        ship=ship,
        takeoff_time_s=takeoff_time_s,
        segments=segments,
    )
    actual_burst = detonation_position(
        release_position,
        UAV_SPEED_MPS * burst_heading,
        delay_s=detonation_delay_s,
    )
    if not np.allclose(actual_burst, burst_center, rtol=0.0, atol=1e-8):
        raise RuntimeError("constructed bomb trajectory misses burst centre")
    return path, release_position, burst_heading


def _merge_intervals(
    intervals: tuple[ClosedInterval, ...],
) -> tuple[ClosedInterval, ...]:
    merged: list[ClosedInterval] = []
    for interval in sorted(intervals):
        if merged and interval.start_s <= merged[-1].end_s + 1e-9:
            merged[-1] = ClosedInterval(
                merged[-1].start_s,
                max(merged[-1].end_s, interval.end_s),
            )
        else:
            merged.append(interval)
    return tuple(merged)


def intersect_interval_sets(
    left: tuple[ClosedInterval, ...],
    right: tuple[ClosedInterval, ...],
) -> tuple[ClosedInterval, ...]:
    intersections = []
    for first in left:
        for second in right:
            start_s = max(first.start_s, second.start_s)
            end_s = min(first.end_s, second.end_s)
            if end_s >= start_s:
                intersections.append(ClosedInterval(start_s, end_s))
    return _merge_intervals(tuple(intersections))


def _maximum_uncovered_interval(
    detection_components: tuple[ClosedInterval, ...],
    covered_intervals: tuple[ClosedInterval, ...],
) -> float:
    maximum = 0.0
    for component in detection_components:
        cursor = component.start_s
        for covered in covered_intervals:
            if covered.end_s <= cursor or covered.start_s >= component.end_s:
                continue
            maximum = max(maximum, max(0.0, covered.start_s - cursor))
            cursor = max(cursor, covered.end_s)
        maximum = max(maximum, max(0.0, component.end_s - cursor))
    return maximum


def smoke_full_coverage_intervals(
    ship: ShipMotion,
    smoke: SmokeCloud,
    *,
    ship_radius_m: float = 80.0,
) -> tuple[ClosedInterval, ...]:
    return intervals_where_nonnegative(
        lambda time_s: -single_smoke_gap(
            ship.position(time_s),
            smoke.burst_center_m,
            smoke.radius(time_s),
            ship_radius_m=ship_radius_m,
        ),
        start_s=smoke.burst_time_s,
        end_s=smoke.failure_time_s,
        max_step_s=0.02,
    )


def evaluate_smoke_against_detection(
    *,
    ship: ShipMotion,
    smoke: SmokeCloud,
    detection: DetectionSet,
) -> tuple[
    tuple[ClosedInterval, ...],
    float,
    float,
    float,
    float,
    CertificationStatus,
]:
    smoke_intervals = smoke_full_coverage_intervals(ship, smoke)
    covered = intersect_interval_sets(detection.components, smoke_intervals)
    covered_duration = sum(interval.duration_s for interval in covered)
    exposed_duration = max(0.0, detection.duration_s - covered_duration)
    maximum_exposed = _maximum_uncovered_interval(detection.components, covered)

    margin_samples = []
    for component in detection.components:
        sample_count = max(3, int(np.ceil(component.duration_s / 0.05)) + 1)
        for time_s in np.linspace(component.start_s, component.end_s, sample_count):
            margin_samples.append(
                -single_smoke_gap(
                    ship.position(float(time_s)),
                    smoke.burst_center_m,
                    smoke.radius(float(time_s)),
                )
            )
    minimum_margin = min(margin_samples, default=float("inf"))
    if exposed_duration > 1e-8:
        strict_status = CertificationStatus.CERTIFIED_INFEASIBLE
    else:
        continuous = certify_single_smoke_continuous_coverage(
            ship_position=ship.position,
            smoke=smoke,
            detection_components=detection.components,
            ship_speed_bound_mps=ship.speed_mps,
        )
        strict_status = continuous.status
    return (
        covered,
        covered_duration,
        exposed_duration,
        maximum_exposed,
        minimum_margin,
        strict_status,
    )


def _candidate_center_times(
    components: tuple[ClosedInterval, ...],
    half_width_s: float,
) -> tuple[float, ...]:
    times: set[float] = set()
    for component in components:
        midpoint = 0.5 * (component.start_s + component.end_s)
        times.update(
            {
                component.start_s,
                component.end_s,
                midpoint,
                component.start_s + half_width_s,
                component.end_s - half_width_s,
            }
        )
    return tuple(sorted(time_s for time_s in times if time_s >= 0.0))


def generate_q1_candidates(
    *,
    ship: ShipMotion,
    detection: DetectionSet,
    uav_available_time_s: float,
    operation_radius_m: float = 12000.0,
    command_time_s: float = 0.0,
    minimum_release_response_s: float = 2.0,
    detonation_delay_s: float = 3.5,
    maximum_smoke_radius_m: float = 120.0,
    ship_radius_m: float = 80.0,
) -> tuple[Q1Candidate, ...]:
    half_width_s = (
        maximum_smoke_radius_m - ship_radius_m
    ) / ship.speed_mps
    candidates: list[Q1Candidate] = []
    takeoff_time_s = uav_available_time_s
    for center_time_s in _candidate_center_times(
        detection.components,
        half_width_s,
    ):
        burst_center = ship.position(center_time_s)
        start = ship.position(takeoff_time_s)
        heading = _unit_direction(burst_center - start)
        release_position = (
            burst_center - UAV_SPEED_MPS * detonation_delay_s * heading
        )
        direct_distance = float(np.linalg.norm(release_position - start))
        earliest_release = max(
            command_time_s + minimum_release_response_s,
            takeoff_time_s + direct_distance / UAV_SPEED_MPS,
        )
        desired_burst = max(
            earliest_release + detonation_delay_s,
            center_time_s - half_width_s,
        )
        release_time_s = desired_burst - detonation_delay_s

        direct_heading = _unit_direction(release_position - start)
        available_length = UAV_SPEED_MPS * (
            release_time_s - takeoff_time_s
        )
        if (
            abs(available_length - direct_distance) <= 1e-7
            and not np.allclose(
                direct_heading,
                heading,
                rtol=0.0,
                atol=1e-8,
            )
        ):
            release_time_s += 0.05
            desired_burst += 0.05
        try:
            path, release_position, burst_heading = (
                build_shipborne_release_path(
                    ship=ship,
                    takeoff_time_s=takeoff_time_s,
                    release_time_s=release_time_s,
                    burst_center_m=burst_center,
                    detonation_delay_s=detonation_delay_s,
                )
            )
            events = BombEvents(
                command_time_s=command_time_s,
                release_time_s=release_time_s,
                burst_time_s=desired_burst,
                minimum_response_s=minimum_release_response_s,
                detonation_delay_s=detonation_delay_s,
            )
        except ValueError:
            continue
        radius_certificate = certify_operation_radius(
            path,
            operation_radius_m=operation_radius_m,
        )
        if radius_certificate.status != "certified_feasible":
            continue
        smoke = SmokeCloud(
            burst_time_s=events.burst_time_s,
            burst_center_m=burst_center,
            maximum_radius_m=maximum_smoke_radius_m,
        )
        (
            covered,
            covered_duration,
            exposed_duration,
            maximum_exposed,
            minimum_margin,
            strict_status,
        ) = evaluate_smoke_against_detection(
            ship=ship,
            smoke=smoke,
            detection=detection,
        )
        candidates.append(
            Q1Candidate(
                strict_status=strict_status,
                covered_duration_s=covered_duration,
                exposed_duration_s=exposed_duration,
                maximum_exposed_interval_s=maximum_exposed,
                minimum_margin_m=minimum_margin,
                flight_distance_m=UAV_SPEED_MPS
                * (release_time_s - takeoff_time_s),
                command_time_s=command_time_s,
                takeoff_time_s=takeoff_time_s,
                release_time_s=release_time_s,
                burst_time_s=events.burst_time_s,
                coverage_center_time_s=center_time_s,
                release_position_m=release_position,
                burst_center_m=burst_center,
                burst_heading=burst_heading,
                covered_intervals=covered,
                path=path,
                smoke=smoke,
                reachability_status=radius_certificate.status,
            )
        )
    return tuple(candidates)
