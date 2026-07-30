import numpy as np

from smoke_defense.coverage import CertificationStatus
from smoke_defense.events import ClosedInterval
from smoke_defense.smoke import SmokeCloud
from smoke_defense.verification import certify_single_smoke_continuous_coverage


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
