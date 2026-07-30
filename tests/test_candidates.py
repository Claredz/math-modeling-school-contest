import numpy as np
import pytest

from smoke_defense.candidates import (
    Q1Candidate,
    build_shipborne_release_path,
    candidate_rank_key,
    generate_q1_candidates,
)
from smoke_defense.coverage import CertificationStatus
from smoke_defense.detection import DetectionSet
from smoke_defense.dynamics import ShipMotion
from smoke_defense.events import ClosedInterval
from smoke_defense.path_constraints import certify_operation_radius
from smoke_defense.smoke import detonation_position


def test_release_path_starts_on_moving_ship_and_reaches_burst_center():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    burst_center = ship.position(18.0)

    release_path, release_position, burst_heading = build_shipborne_release_path(
        ship=ship,
        takeoff_time_s=0.0,
        release_time_s=2.0,
        burst_center_m=burst_center,
        detonation_delay_s=3.5,
    )

    assert release_path.position(0.0) == pytest.approx(ship.position(0.0))
    assert release_path.position(2.0) == pytest.approx(release_position)
    assert np.linalg.norm(release_path.velocity(2.0)) == pytest.approx(28.0)
    assert detonation_position(
        release_position,
        28.0 * burst_heading,
        delay_s=3.5,
    ) == pytest.approx(burst_center)


def test_lexicographic_rank_prefers_coverage_before_flight_distance():
    short_flight = Q1Candidate.for_ranking_test(
        strict_status=CertificationStatus.CERTIFIED_INFEASIBLE,
        covered_duration_s=8.0,
        maximum_exposed_interval_s=4.0,
        minimum_margin_m=-20.0,
        flight_distance_m=50.0,
    )
    long_flight = Q1Candidate.for_ranking_test(
        strict_status=CertificationStatus.CERTIFIED_INFEASIBLE,
        covered_duration_s=9.0,
        maximum_exposed_interval_s=4.0,
        minimum_margin_m=-20.0,
        flight_distance_m=500.0,
    )

    assert candidate_rank_key(long_flight) > candidate_rank_key(short_flight)


def test_lexicographic_rank_ignores_sub_nanosecond_duration_noise():
    noisy_longer = Q1Candidate.for_ranking_test(
        strict_status=CertificationStatus.CERTIFIED_INFEASIBLE,
        covered_duration_s=9.0 + 1e-12,
        maximum_exposed_interval_s=8.0,
        minimum_margin_m=-20.0,
        flight_distance_m=50.0,
    )
    better_secondary = Q1Candidate.for_ranking_test(
        strict_status=CertificationStatus.CERTIFIED_INFEASIBLE,
        covered_duration_s=9.0,
        maximum_exposed_interval_s=4.0,
        minimum_margin_m=-20.0,
        flight_distance_m=500.0,
    )

    assert candidate_rank_key(better_secondary) > candidate_rank_key(noisy_longer)


def test_delayed_takeoff_can_preserve_coverage_and_shorten_flight():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    detection = DetectionSet(
        components=(ClosedInterval(20.0, 30.0),),
        source_events=(),
    )

    candidates = generate_q1_candidates(
        ship=ship,
        detection=detection,
        uav_available_time_s=0.0,
    )
    centered = [
        candidate
        for candidate in candidates
        if candidate.coverage_center_time_s == pytest.approx(25.0)
    ]
    earliest = next(
        candidate
        for candidate in centered
        if candidate.takeoff_time_s == pytest.approx(0.0)
    )
    delayed = [
        candidate for candidate in centered if candidate.takeoff_time_s > 0.0
    ]

    assert delayed
    best_delayed = max(delayed, key=candidate_rank_key)
    assert best_delayed.covered_duration_s == pytest.approx(
        earliest.covered_duration_s
    )
    assert best_delayed.flight_distance_m < earliest.flight_distance_m
    assert best_delayed.path is not None
    assert best_delayed.path.position(
        best_delayed.takeoff_time_s
    ) == pytest.approx(ship.position(best_delayed.takeoff_time_s))
    assert best_delayed.takeoff_time_s < best_delayed.release_time_s
    assert best_delayed.flight_distance_m == pytest.approx(
        sum(
            segment.speed_mps * segment.duration_s
            for segment in best_delayed.path.segments
        )
    )
    assert (
        certify_operation_radius(
            best_delayed.path,
            operation_radius_m=12000.0,
        ).status
        == "certified_feasible"
    )


def test_custom_smoke_timing_changes_candidate_coverage():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    detection = DetectionSet(
        components=(ClosedInterval(10.0, 25.0),),
        source_events=(),
    )

    nominal = generate_q1_candidates(
        ship=ship,
        detection=detection,
        uav_available_time_s=0.0,
    )
    custom = generate_q1_candidates(
        ship=ship,
        detection=detection,
        uav_available_time_s=0.0,
        maximum_smoke_radius_m=130.0,
        smoke_hold_duration_s=1.0,
        smoke_decay_duration_s=1.0,
    )

    assert custom
    assert all(candidate.smoke is not None for candidate in custom)
    assert all(candidate.smoke.maximum_radius_m == 130.0 for candidate in custom)
    assert all(candidate.smoke.hold_duration_s == 1.0 for candidate in custom)
    assert all(candidate.smoke.decay_duration_s == 1.0 for candidate in custom)
    assert max(candidate.covered_duration_s for candidate in custom) < max(
        candidate.covered_duration_s for candidate in nominal
    )


def test_custom_ship_radius_is_used_by_candidate_coverage():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    detection = DetectionSet(
        components=(ClosedInterval(10.0, 30.0),),
        source_events=(),
    )

    candidates = generate_q1_candidates(
        ship=ship,
        detection=detection,
        uav_available_time_s=0.0,
        maximum_smoke_radius_m=120.0,
        ship_radius_m=100.0,
    )

    assert candidates
    assert max(candidate.covered_duration_s for candidate in candidates) <= (
        2.0 * (120.0 - 100.0) / ship.speed_mps + 1e-6
    )


def test_custom_smoke_lifetime_adds_balanced_coverage_center():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    detection = DetectionSet(
        components=(ClosedInterval(10.0, 25.0),),
        source_events=(),
    )

    candidates = generate_q1_candidates(
        ship=ship,
        detection=detection,
        uav_available_time_s=0.0,
        maximum_smoke_radius_m=130.0,
        smoke_hold_duration_s=1.0,
        smoke_decay_duration_s=1.0,
    )
    best = max(candidates, key=candidate_rank_key)

    assert best.covered_intervals
    covered = best.covered_intervals[0]
    left_exposure = covered.start_s - detection.components[0].start_s
    right_exposure = detection.components[0].end_s - covered.end_s
    assert left_exposure == pytest.approx(right_exposure, abs=1e-6)
