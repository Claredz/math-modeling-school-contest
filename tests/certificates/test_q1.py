import pytest

from smoke_defense.certificates.q1 import (
    certify_earliest_smoke_availability,
    certify_single_smoke_duration,
)
from smoke_defense.coverage import CertificationStatus
from smoke_defense.events import ClosedInterval


def test_long_component_is_certified_infeasible():
    result = certify_single_smoke_duration((ClosedInterval(0.0, 11.0),))

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.limit_s == pytest.approx(80.0 / 7.71)


def test_short_components_are_not_certified_feasible():
    result = certify_single_smoke_duration(
        (
            ClosedInterval(0.0, 5.0),
            ClosedInterval(8.0, 13.0),
        )
    )

    assert result.status is CertificationStatus.INDETERMINATE


def test_detection_before_earliest_smoke_is_unavoidably_exposed():
    result = certify_earliest_smoke_availability(
        (ClosedInterval(0.0, 8.0),),
        command_time_s=0.0,
    )

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.earliest_burst_time_s == pytest.approx(5.5)
    assert result.unavoidable_exposure_s == pytest.approx(5.5)


def test_detection_starting_at_earliest_burst_is_not_claimed_feasible():
    result = certify_earliest_smoke_availability(
        (ClosedInterval(5.5, 8.0),),
        command_time_s=0.0,
    )

    assert result.status is CertificationStatus.INDETERMINATE
    assert result.unavoidable_exposure_s == pytest.approx(0.0)
