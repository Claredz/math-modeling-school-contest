import numpy as np
import pytest

from smoke_defense.smoke import SmokeCloud, detonation_position


def test_smoke_radius_boundaries_are_exact():
    smoke = SmokeCloud(
        burst_time_s=10.0,
        burst_center_m=np.array([5.0, -2.0]),
    )

    assert smoke.radius(9.999) == 0.0
    assert smoke.radius(10.0) == pytest.approx(120.0)
    assert smoke.radius(28.0) == pytest.approx(120.0)
    assert smoke.radius(33.0) == pytest.approx(0.0)
    assert smoke.radius(33.001) == 0.0


def test_nominal_inertial_bomb_flight_is_98_metres():
    release = np.array([10.0, 20.0])
    velocity = np.array([0.0, 28.0])

    burst = detonation_position(release, velocity, delay_s=3.5)

    assert np.linalg.norm(burst - release) == pytest.approx(98.0)


def test_smoke_center_is_fixed_after_burst():
    smoke = SmokeCloud(
        burst_time_s=10.0,
        burst_center_m=np.array([5.0, -2.0]),
    )

    assert smoke.center(10.0) == pytest.approx([5.0, -2.0])
    assert smoke.center(32.0) == pytest.approx([5.0, -2.0])
