import numpy as np

from smoke_defense.coverage import CertificationStatus
from smoke_defense.events import ClosedInterval
from smoke_defense.smoke import SmokeCloud
from smoke_defense.verification import (
    certify_multi_smoke_coverage,
    certify_single_smoke_continuous_coverage,
)


def stationary_ship(_time_s: float) -> np.ndarray:
    return np.zeros(2)


def test_continuous_single_smoke_coverage_is_certified():
    smoke = SmokeCloud(
        burst_time_s=0.0,
        burst_center_m=np.array([0.0, 0.0]),
    )

    result = certify_single_smoke_continuous_coverage(
        ship_position=stationary_ship,
        smoke=smoke,
        detection_components=(ClosedInterval(0.0, 10.0),),
    )

    assert result.status is CertificationStatus.CERTIFIED_FEASIBLE


def test_preburst_detection_is_certified_infeasible():
    smoke = SmokeCloud(
        burst_time_s=5.5,
        burst_center_m=np.array([0.0, 0.0]),
    )

    result = certify_single_smoke_continuous_coverage(
        ship_position=stationary_ship,
        smoke=smoke,
        detection_components=(ClosedInterval(0.0, 10.0),),
    )

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.witness_time_s == 0.0


def test_decay_endpoint_is_checked_as_closed_detection_time():
    smoke = SmokeCloud(
        burst_time_s=0.0,
        burst_center_m=np.array([0.0, 0.0]),
    )

    result = certify_single_smoke_continuous_coverage(
        ship_position=stationary_ship,
        smoke=smoke,
        detection_components=(ClosedInterval(22.0, 23.0),),
    )

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.witness_time_s is not None


def test_continuous_spatial_union_succeeds_when_each_smoke_fails_alone():
    smokes = (
        SmokeCloud(
            burst_time_s=0.0,
            burst_center_m=np.array([-50.0, 0.0]),
            maximum_radius_m=110.0,
        ),
        SmokeCloud(
            burst_time_s=0.0,
            burst_center_m=np.array([50.0, 0.0]),
            maximum_radius_m=110.0,
        ),
    )
    component = (ClosedInterval(0.0, 5.0),)
    assert all(
        certify_single_smoke_continuous_coverage(
            ship_position=stationary_ship,
            smoke=smoke,
            detection_components=component,
        ).status
        is CertificationStatus.CERTIFIED_INFEASIBLE
        for smoke in smokes
    )

    result = certify_multi_smoke_coverage(
        ship_position=stationary_ship,
        smokes=smokes,
        detection_components=component,
        ship_speed_bound_mps=0.0,
    )

    assert result.status is CertificationStatus.CERTIFIED_FEASIBLE
    assert result.maximum_gap_upper_bound_m <= 0.0
    assert result.minimum_margin_m > 0.0


def test_multi_smoke_preburst_closed_endpoint_is_infeasible():
    smokes = (
        SmokeCloud(5.5, np.array([-50.0, 0.0]), maximum_radius_m=110.0),
        SmokeCloud(5.5, np.array([50.0, 0.0]), maximum_radius_m=110.0),
    )

    result = certify_multi_smoke_coverage(
        ship_position=stationary_ship,
        smokes=smokes,
        detection_components=(ClosedInterval(0.0, 10.0),),
        ship_speed_bound_mps=0.0,
    )

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.witness_time_s == 0.0
    assert result.witness_m is not None


def test_burst_event_uses_post_jump_radius_under_closed_semantics():
    smokes = (
        SmokeCloud(5.5, np.array([-50.0, 0.0]), maximum_radius_m=110.0),
        SmokeCloud(5.5, np.array([50.0, 0.0]), maximum_radius_m=110.0),
    )

    result = certify_multi_smoke_coverage(
        ship_position=stationary_ship,
        smokes=smokes,
        detection_components=(ClosedInterval(5.5, 6.0),),
        ship_speed_bound_mps=0.0,
    )

    assert result.status is CertificationStatus.CERTIFIED_FEASIBLE
    assert 5.5 in result.checked_event_times_s


def test_hold_end_and_failure_are_checked_as_separate_events():
    smokes = (
        SmokeCloud(0.0, np.array([-50.0, 0.0]), maximum_radius_m=110.0),
        SmokeCloud(0.0, np.array([50.0, 0.0]), maximum_radius_m=110.0),
    )

    result = certify_multi_smoke_coverage(
        ship_position=stationary_ship,
        smokes=smokes,
        detection_components=(ClosedInterval(17.0, 23.0),),
        ship_speed_bound_mps=0.0,
    )

    assert 18.0 in result.checked_event_times_s
    assert 23.0 in result.checked_event_times_s
    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.witness_time_s == 23.0


def test_continuous_tangent_case_is_indeterminate_at_time_tolerance():
    smoke = SmokeCloud(
        burst_time_s=0.0,
        burst_center_m=np.zeros(2),
        maximum_radius_m=80.0,
    )

    result = certify_multi_smoke_coverage(
        ship_position=stationary_ship,
        smokes=(smoke,),
        detection_components=(ClosedInterval(0.0, 1.0),),
        ship_speed_bound_mps=1.0,
        time_tolerance_s=0.1,
    )

    assert result.status is CertificationStatus.INDETERMINATE
    assert result.unresolved_intervals


def test_sub_tolerance_positive_event_gap_is_still_infeasible():
    angle_rad = np.deg2rad(11.25)
    axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    smokes = (
        SmokeCloud(0.0, -30.0 * axis, maximum_radius_m=85.43),
        SmokeCloud(0.0, 30.0 * axis, maximum_radius_m=85.43),
    )

    result = certify_multi_smoke_coverage(
        ship_position=stationary_ship,
        smokes=smokes,
        detection_components=(ClosedInterval(0.0, 0.0),),
        ship_speed_bound_mps=0.0,
        spatial_tolerance_m=0.05,
    )

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.witness_time_s == 0.0
