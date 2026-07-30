import numpy as np
import pytest

from smoke_defense.candidates import (
    Q1Candidate,
    build_shipborne_release_path,
    candidate_rank_key,
)
from smoke_defense.coverage import CertificationStatus
from smoke_defense.dynamics import ShipMotion
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
