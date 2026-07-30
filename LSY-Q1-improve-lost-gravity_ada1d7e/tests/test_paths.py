import numpy as np
import pytest

from smoke_defense.dynamics import ShipMotion
from smoke_defense.paths import LinearFlightSegment, ShipborneUavPath


def make_path() -> ShipborneUavPath:
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=5.0)
    takeoff_position = ship.position(5.0)
    first_end = takeoff_position + np.array([0.0, 280.0])
    second_end = first_end + np.array([280.0, 0.0])
    return ShipborneUavPath(
        ship=ship,
        takeoff_time_s=5.0,
        segments=(
            LinearFlightSegment(5.0, 15.0, takeoff_position, first_end),
            LinearFlightSegment(15.0, 25.0, first_end, second_end),
        ),
    )


def test_uav_waits_on_moving_ship_before_takeoff():
    path = make_path()

    assert path.position(3.0) == pytest.approx(path.ship.position(3.0))


def test_launch_position_equals_ship_at_takeoff():
    path = make_path()

    assert path.position(5.0) == pytest.approx(path.ship.position(5.0))


def test_all_airborne_segments_have_fixed_speed():
    path = make_path()

    assert all(segment.speed_mps == pytest.approx(28.0) for segment in path.segments)


def test_path_rejects_non_shipborne_launch_position():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=5.0)
    with pytest.raises(ValueError, match="ship position"):
        ShipborneUavPath(
            ship=ship,
            takeoff_time_s=5.0,
            segments=(
                LinearFlightSegment(
                    5.0,
                    6.0,
                    np.array([26.0, 0.0]),
                    np.array([54.0, 0.0]),
                ),
            ),
        )


def test_path_rejects_airborne_hovering_or_wrong_speed():
    with pytest.raises(ValueError, match="28"):
        LinearFlightSegment(
            0.0,
            1.0,
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )


def test_piecewise_path_rejects_discontinuous_nodes():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=0.0)
    with pytest.raises(ValueError, match="continuous"):
        ShipborneUavPath(
            ship=ship,
            takeoff_time_s=0.0,
            segments=(
                LinearFlightSegment(
                    0.0,
                    1.0,
                    np.array([0.0, 0.0]),
                    np.array([28.0, 0.0]),
                ),
                LinearFlightSegment(
                    1.0,
                    2.0,
                    np.array([29.0, 0.0]),
                    np.array([57.0, 0.0]),
                ),
            ),
        )
