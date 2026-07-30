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

RANKING_TIME_DECIMALS = 9


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
        round(candidate.covered_duration_s, RANKING_TIME_DECIMALS),
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
        lipschitz_bound_per_s=ship.speed_mps + smoke.decay_rate_mps,
    )


def _minimum_smoke_margin(
    *,
    ship: ShipMotion,
    smoke: SmokeCloud,
    detection: DetectionSet,
    ship_radius_m: float,
) -> float:
    """Return the exact margin infimum over the detection components.

    Within each smoke phase the radius is affine and the negative distance
    from a constant-velocity ship to the fixed centre is concave. The minimum
    therefore lies at a phase endpoint. The pre-burst left limit is checked
    separately because the radius jumps at burst.
    """

    margins: list[float] = []

    def margin(time_s: float, *, radius_m: float | None = None) -> float:
        smoke_radius_m = (
            smoke.radius(time_s) if radius_m is None else radius_m
        )
        return -single_smoke_gap(
            ship.position(time_s),
            smoke.burst_center_m,
            smoke_radius_m,
            ship_radius_m=ship_radius_m,
        )

    critical_times = (
        smoke.burst_time_s,
        smoke.hold_end_time_s,
        smoke.failure_time_s,
    )
    for component in detection.components:
        times = {component.start_s, component.end_s}
        times.update(
            time_s
            for time_s in critical_times
            if component.start_s <= time_s <= component.end_s
        )
        margins.extend(margin(time_s) for time_s in times)
        if (
            component.start_s < smoke.burst_time_s
            <= component.end_s
        ):
            margins.append(
                margin(smoke.burst_time_s, radius_m=0.0)
            )
    return min(margins, default=float("inf"))


def evaluate_smoke_against_detection(
    *,
    ship: ShipMotion,
    smoke: SmokeCloud,
    detection: DetectionSet,
    ship_radius_m: float = 80.0,
) -> tuple[
    tuple[ClosedInterval, ...],
    float,
    float,
    float,
    float,
    CertificationStatus,
]:
    smoke_intervals = smoke_full_coverage_intervals(
        ship,
        smoke,
        ship_radius_m=ship_radius_m,
    )
    covered = intersect_interval_sets(detection.components, smoke_intervals)
    covered_duration = sum(interval.duration_s for interval in covered)
    exposed_duration = max(0.0, detection.duration_s - covered_duration)
    maximum_exposed = _maximum_uncovered_interval(detection.components, covered)

    minimum_margin = _minimum_smoke_margin(
        ship=ship,
        smoke=smoke,
        detection=detection,
        ship_radius_m=ship_radius_m,
    )
    if exposed_duration > 1e-8:
        strict_status = CertificationStatus.CERTIFIED_INFEASIBLE
    else:
        continuous = certify_single_smoke_continuous_coverage(
            ship_position=ship.position,
            smoke=smoke,
            detection_components=detection.components,
            ship_radius_m=ship_radius_m,
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
    coverage_start_offset_s: float,
    coverage_end_offset_s: float,
) -> tuple[float, ...]:
    times: set[float] = set()
    coverage_midpoint_offset_s = 0.5 * (
        coverage_start_offset_s + coverage_end_offset_s
    )
    for component in components:
        midpoint = 0.5 * (component.start_s + component.end_s)
        times.update(
            {
                component.start_s,
                component.end_s,
                midpoint - coverage_midpoint_offset_s,
                component.start_s - coverage_start_offset_s,
                component.end_s - coverage_end_offset_s,
            }
        )
    return tuple(sorted(time_s for time_s in times if time_s >= 0.0))


def _ideal_smoke_coverage_offsets(
    *,
    ship_speed_mps: float,
    maximum_smoke_radius_m: float,
    smoke_hold_duration_s: float,
    smoke_decay_duration_s: float,
    ship_radius_m: float,
) -> tuple[float, float]:
    """Return ideal full-coverage offsets around a spatial centre time."""

    if maximum_smoke_radius_m < ship_radius_m:
        return 0.0, 0.0
    geometric_half_width_s = (
        maximum_smoke_radius_m - ship_radius_m
    ) / ship_speed_mps
    canonical_ship = ShipMotion(
        (0.0, 0.0),
        heading_rad=0.0,
        speed_mps=ship_speed_mps,
    )
    canonical_smoke = SmokeCloud(
        burst_time_s=-geometric_half_width_s,
        burst_center_m=np.zeros(2),
        maximum_radius_m=maximum_smoke_radius_m,
        hold_duration_s=smoke_hold_duration_s,
        decay_duration_s=smoke_decay_duration_s,
    )
    intervals = smoke_full_coverage_intervals(
        canonical_ship,
        canonical_smoke,
        ship_radius_m=ship_radius_m,
    )
    if not intervals:
        return 0.0, 0.0
    longest = max(intervals, key=lambda interval: interval.duration_s)
    return longest.start_s, longest.end_s


def _latest_reachable_takeoff_time(
    *,
    ship: ShipMotion,
    uav_available_time_s: float,
    release_time_s: float,
    release_position_m: np.ndarray,
) -> float | None:
    """Return the latest takeoff that can still reach a fixed release event.

    UAV speed exceeds ship speed, so reachability slack is strictly decreasing
    in takeoff time. A reachable interval therefore has one latest boundary.
    """

    if release_time_s <= uav_available_time_s:
        return None

    def reachability_slack(time_s: float) -> float:
        available_distance = UAV_SPEED_MPS * (release_time_s - time_s)
        required_distance = float(
            np.linalg.norm(release_position_m - ship.position(time_s))
        )
        return available_distance - required_distance

    if reachability_slack(uav_available_time_s) < -1e-9:
        return None
    latest_query_time_s = release_time_s - 1e-9
    if reachability_slack(latest_query_time_s) >= 0.0:
        boundary_time_s = latest_query_time_s
    else:
        boundary_time_s = float(
            brentq(
                reachability_slack,
                uav_available_time_s,
                latest_query_time_s,
                xtol=1e-12,
            )
        )
    return max(
        uav_available_time_s,
        boundary_time_s - 1e-6,
    )


def _candidate_takeoff_times(
    *,
    ship: ShipMotion,
    uav_available_time_s: float,
    command_time_s: float,
    minimum_release_response_s: float,
    detonation_delay_s: float,
    center_time_s: float,
    burst_center_m: np.ndarray,
    coverage_half_width_s: float,
) -> tuple[float, ...]:
    """Include earliest availability and the latest coverage-preserving launch."""

    times = {uav_available_time_s}
    target_burst_time_s = max(
        center_time_s - coverage_half_width_s,
        command_time_s
        + minimum_release_response_s
        + detonation_delay_s,
    )
    target_release_time_s = target_burst_time_s - detonation_delay_s
    if not (
        target_release_time_s > uav_available_time_s
        and target_release_time_s < center_time_s
    ):
        return tuple(sorted(times))

    burst_heading = _unit_direction(
        burst_center_m - ship.position(target_release_time_s)
    )
    release_position = (
        burst_center_m
        - UAV_SPEED_MPS * detonation_delay_s * burst_heading
    )
    latest = _latest_reachable_takeoff_time(
        ship=ship,
        uav_available_time_s=uav_available_time_s,
        release_time_s=target_release_time_s,
        release_position_m=release_position,
    )
    if latest is not None and latest > uav_available_time_s + 1e-9:
        times.add(latest)
    return tuple(sorted(times))


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
    smoke_hold_duration_s: float = 18.0,
    smoke_decay_duration_s: float = 5.0,
    ship_radius_m: float = 80.0,
) -> tuple[Q1Candidate, ...]:
    half_width_s = max(
        0.0,
        (maximum_smoke_radius_m - ship_radius_m) / ship.speed_mps,
    )
    coverage_start_offset_s, coverage_end_offset_s = (
        _ideal_smoke_coverage_offsets(
            ship_speed_mps=ship.speed_mps,
            maximum_smoke_radius_m=maximum_smoke_radius_m,
            smoke_hold_duration_s=smoke_hold_duration_s,
            smoke_decay_duration_s=smoke_decay_duration_s,
            ship_radius_m=ship_radius_m,
        )
    )
    candidates: list[Q1Candidate] = []
    for center_time_s in _candidate_center_times(
        detection.components,
        coverage_start_offset_s,
        coverage_end_offset_s,
    ):
        burst_center = ship.position(center_time_s)
        for takeoff_time_s in _candidate_takeoff_times(
            ship=ship,
            uav_available_time_s=uav_available_time_s,
            command_time_s=command_time_s,
            minimum_release_response_s=minimum_release_response_s,
            detonation_delay_s=detonation_delay_s,
            center_time_s=center_time_s,
            burst_center_m=burst_center,
            coverage_half_width_s=half_width_s,
        ):
            start = ship.position(takeoff_time_s)
            heading = _unit_direction(burst_center - start)
            release_position = (
                burst_center
                - UAV_SPEED_MPS * detonation_delay_s * heading
            )
            direct_distance = float(
                np.linalg.norm(release_position - start)
            )
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
                hold_duration_s=smoke_hold_duration_s,
                decay_duration_s=smoke_decay_duration_s,
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
                ship_radius_m=ship_radius_m,
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
