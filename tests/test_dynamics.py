import numpy as np
import pytest

from smoke_defense.dynamics import (
    MissileGuidanceSpec,
    ShipMotion,
    inertial_heading_rate,
    inertial_pursuit_rhs,
    initial_inertial_state,
    integrate_inertial_missile,
    integrate_instantaneous_reference,
)


@pytest.fixture
def ship() -> ShipMotion:
    return ShipMotion(
        initial_position_m=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=7.71,
    )


@pytest.fixture
def nominal_spec() -> MissileGuidanceSpec:
    return MissileGuidanceSpec(
        speed_mps=320.0,
        heading_response_rate_per_s=1.0,
        max_turn_rate_deg_s=10.0,
    )


def test_ship_motion_is_uniform_and_straight(ship):
    np.testing.assert_allclose(ship.position(10.0), [77.1, 0.0], atol=1e-12)


def test_initial_heading_points_to_ship(ship):
    state = initial_inertial_state(
        missile_position_m=(0.0, 1000.0),
        ship_position_m=ship.position(0.0),
    )

    np.testing.assert_allclose(state[:2], [0.0, 1000.0], atol=1e-12)
    assert state[2] == pytest.approx(-np.pi / 2)


def test_missile_speed_magnitude_is_constant(ship, nominal_spec):
    state = initial_inertial_state(
        missile_position_m=(10000.0, 0.0),
        ship_position_m=ship.position(0.0),
    )
    rhs = inertial_pursuit_rhs(0.0, state, ship.position, nominal_spec)

    assert np.linalg.norm(rhs[:2]) == pytest.approx(320.0)


def test_heading_rate_is_clipped(nominal_spec):
    rate = inertial_heading_rate(
        line_of_sight_rad=np.pi / 2,
        heading_rad=0.0,
        spec=nominal_spec,
    )

    assert rate == pytest.approx(np.deg2rad(10.0))


def test_heading_rate_is_first_order_when_unsaturated():
    spec = MissileGuidanceSpec(
        speed_mps=320.0,
        heading_response_rate_per_s=0.5,
        max_turn_rate_deg_s=180.0,
    )

    rate = inertial_heading_rate(
        line_of_sight_rad=0.2,
        heading_rad=0.0,
        spec=spec,
    )

    assert rate == pytest.approx(0.1)


def test_initial_hit_short_circuits_undefined_line_of_sight(ship, nominal_spec):
    trajectory = integrate_inertial_missile(
        initial_position_m=(80.0, 0.0),
        appearance_time_s=0.0,
        ship_position=ship.position,
        spec=nominal_spec,
        t_final_s=20.0,
        hit_radius_m=80.0,
    )

    assert trajectory.hit_time_s == pytest.approx(0.0)
    np.testing.assert_allclose(trajectory.position(0.0), [80.0, 0.0])


def test_collinear_inertial_pursuit_matches_closing_speed(ship, nominal_spec):
    trajectory = integrate_inertial_missile(
        initial_position_m=(10000.0, 0.0),
        appearance_time_s=0.0,
        ship_position=ship.position,
        spec=nominal_spec,
        t_final_s=40.0,
        hit_radius_m=80.0,
    )
    expected_hit_time = (10000.0 - 80.0) / (320.0 + 7.71)

    assert trajectory.hit_time_s == pytest.approx(expected_hit_time, abs=1e-7)


def test_large_response_parameters_approach_instantaneous_reference(ship):
    initial_position = (0.0, 10000.0)
    slow = MissileGuidanceSpec(
        speed_mps=320.0,
        heading_response_rate_per_s=0.5,
        max_turn_rate_deg_s=5.0,
    )
    fast = MissileGuidanceSpec(
        speed_mps=320.0,
        heading_response_rate_per_s=20.0,
        max_turn_rate_deg_s=180.0,
    )
    slow_trajectory = integrate_inertial_missile(
        initial_position,
        0.0,
        ship.position,
        slow,
        t_final_s=10.0,
    )
    fast_trajectory = integrate_inertial_missile(
        initial_position,
        0.0,
        ship.position,
        fast,
        t_final_s=10.0,
    )
    reference = integrate_instantaneous_reference(
        initial_position,
        0.0,
        ship.position,
        speed_mps=320.0,
        t_final_s=10.0,
    )
    sample_times = np.linspace(0.0, 10.0, 101)
    slow_error = max(
        np.linalg.norm(slow_trajectory.position(t) - reference.position(t))
        for t in sample_times
    )
    fast_error = max(
        np.linalg.norm(fast_trajectory.position(t) - reference.position(t))
        for t in sample_times
    )

    assert fast_error < slow_error
