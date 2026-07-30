import numpy as np
import pytest

from smoke_defense.dynamics import ShipMotion
from smoke_defense.path_constraints import (
    certify_operation_radius,
    certify_pairwise_separation,
)
from smoke_defense.paths import LinearFlightSegment, ShipborneUavPath


def straight_path(
    *,
    ship: ShipMotion,
    takeoff_time_s: float,
    heading: np.ndarray,
    duration_s: float,
) -> ShipborneUavPath:
    start = ship.position(takeoff_time_s)
    unit_heading = heading / np.linalg.norm(heading)
    end = start + 28.0 * duration_s * unit_heading
    return ShipborneUavPath(
        ship=ship,
        takeoff_time_s=takeoff_time_s,
        segments=(
            LinearFlightSegment(
                takeoff_time_s,
                takeoff_time_s + duration_s,
                start,
                end,
            ),
        ),
    )


def test_simultaneous_colocated_launches_fail_safe_distance():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=5.0)
    path_a = straight_path(
        ship=ship,
        takeoff_time_s=2.0,
        heading=np.array([1.0, 0.0]),
        duration_s=10.0,
    )
    path_b = straight_path(
        ship=ship,
        takeoff_time_s=2.0,
        heading=np.array([0.0, 1.0]),
        duration_s=10.0,
    )

    certificate = certify_pairwise_separation(
        path_a,
        path_b,
        safe_distance_m=100.0,
    )

    assert certificate.status == "certified_infeasible"
    assert certificate.minimum_value == pytest.approx(0.0)


def test_uav_still_on_ship_is_excluded_from_airborne_separation():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0)
    early = straight_path(
        ship=ship,
        takeoff_time_s=0.0,
        heading=np.array([1.0, 0.0]),
        duration_s=10.0,
    )
    late = straight_path(
        ship=ship,
        takeoff_time_s=5.0,
        heading=np.array([0.0, 1.0]),
        duration_s=5.0,
    )

    certificate = certify_pairwise_separation(
        early,
        late,
        safe_distance_m=100.0,
    )

    assert certificate.status == "certified_feasible"
    assert certificate.minimum_value == pytest.approx(140.0)
    assert certificate.critical_time_s == pytest.approx(5.0)


def test_pairwise_closest_point_checks_segment_interior():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0)
    first_start = ship.position(0.0)
    first_turn = first_start + np.array([56.0, 0.0])
    first = ShipborneUavPath(
        ship=ship,
        takeoff_time_s=0.0,
        segments=(
            LinearFlightSegment(0.0, 2.0, first_start, first_turn),
            LinearFlightSegment(
                2.0,
                4.0,
                first_turn,
                first_turn + np.array([-56.0, 0.0]),
            ),
        ),
    )
    start = ship.position(1.0)
    second_heading = np.array([np.sqrt(3.0) / 2.0, 0.5])
    second = ShipborneUavPath(
        ship=ship,
        takeoff_time_s=1.0,
        segments=(
            LinearFlightSegment(
                1.0,
                4.0,
                start,
                start + 84.0 * second_heading,
            ),
        ),
    )

    certificate = certify_pairwise_separation(
        first,
        second,
        safe_distance_m=20.0,
    )

    assert certificate.status == "certified_feasible"
    assert certificate.minimum_value < 28.0
    assert 2.0 < certificate.critical_time_s < 3.0


def test_operation_radius_uses_moving_ship_and_exact_segment_endpoint_maximum():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=10.0)
    path = straight_path(
        ship=ship,
        takeoff_time_s=0.0,
        heading=np.array([-1.0, 0.0]),
        duration_s=2.0,
    )

    feasible = certify_operation_radius(path, operation_radius_m=76.0)
    infeasible = certify_operation_radius(path, operation_radius_m=75.0)

    assert feasible.status == "certified_feasible"
    assert feasible.maximum_value == pytest.approx(76.0)
    assert infeasible.status == "certified_infeasible"
