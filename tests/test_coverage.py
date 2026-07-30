import numpy as np
import pytest

from smoke_defense.coverage import (
    CertificationStatus,
    Disk,
    certify_union_coverage,
    evaluate_union_gap_at_time,
    single_smoke_gap,
)


def test_single_smoke_uses_exact_disk_containment():
    gap = single_smoke_gap(
        ship_center_m=np.array([0.0, 0.0]),
        smoke_center_m=np.array([30.0, 0.0]),
        smoke_radius_m=120.0,
    )

    assert gap == pytest.approx(-10.0)


def test_two_smokes_can_cover_when_neither_covers_alone():
    ship = Disk(np.array([0.0, 0.0]), 80.0)
    left = Disk(np.array([-30.0, 0.0]), 86.0)
    right = Disk(np.array([30.0, 0.0]), 86.0)

    assert single_smoke_gap(
        ship.center_m,
        left.center_m,
        left.radius_m,
        ship_radius_m=ship.radius_m,
    ) > 0
    assert single_smoke_gap(
        ship.center_m,
        right.center_m,
        right.radius_m,
        ship_radius_m=ship.radius_m,
    ) > 0

    result = certify_union_coverage(ship, (left, right))

    assert result.status is CertificationStatus.CERTIFIED_FEASIBLE


def test_union_coverage_finds_exact_uncovered_witness():
    ship = Disk(np.array([0.0, 0.0]), 80.0)
    smokes = (
        Disk(np.array([-60.0, 0.0]), 50.0),
        Disk(np.array([60.0, 0.0]), 50.0),
    )

    result = certify_union_coverage(ship, smokes)

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.witness_m is not None
    assert np.linalg.norm(result.witness_m) <= 80.0


def test_empty_smoke_union_is_infeasible():
    ship = Disk(np.array([0.0, 0.0]), 80.0)

    result = certify_union_coverage(ship, ())

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE


def test_union_gap_is_nonpositive_only_through_joint_spatial_coverage():
    ship = Disk(np.array([0.0, 0.0]), 80.0)
    smokes = (
        Disk(np.array([-50.0, 0.0]), 110.0),
        Disk(np.array([50.0, 0.0]), 110.0),
    )
    assert all(
        single_smoke_gap(
            ship.center_m,
            smoke.center_m,
            smoke.radius_m,
            ship_radius_m=ship.radius_m,
        )
        > 0
        for smoke in smokes
    )

    result = evaluate_union_gap_at_time(
        ship,
        smokes,
        spatial_tolerance_m=0.05,
    )

    assert result.status is CertificationStatus.CERTIFIED_FEASIBLE
    assert result.upper_bound_m <= 0.0
    assert result.spatial_error_m <= 0.05


def test_union_gap_returns_positive_bound_and_exact_witness():
    ship = Disk(np.array([0.0, 0.0]), 80.0)
    smokes = (
        Disk(np.array([-60.0, 0.0]), 50.0),
        Disk(np.array([60.0, 0.0]), 50.0),
    )

    result = evaluate_union_gap_at_time(
        ship,
        smokes,
        spatial_tolerance_m=0.05,
    )

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.lower_bound_m > 0.0
    assert result.witness_m is not None
    assert np.linalg.norm(result.witness_m - ship.center_m) <= ship.radius_m
    assert all(
        np.linalg.norm(result.witness_m - smoke.center_m) > smoke.radius_m
        for smoke in smokes
    )


def test_single_concentric_smoke_gap_uses_boundary_witness():
    ship = Disk(np.array([0.0, 0.0]), 80.0)
    smoke = Disk(np.array([0.0, 0.0]), 70.0)

    result = evaluate_union_gap_at_time(ship, (smoke,))

    assert result.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.witness_m is not None
    assert np.linalg.norm(result.witness_m - ship.center_m) == pytest.approx(80.0)
    assert np.linalg.norm(result.witness_m - smoke.center_m) > smoke.radius_m


def test_union_gap_low_polygon_cap_is_indeterminate_not_false_success():
    ship = Disk(np.array([0.0, 0.0]), 80.0)
    smokes = (
        Disk(np.array([-30.0, 0.0]), 86.0),
        Disk(np.array([30.0, 0.0]), 86.0),
    )

    result = evaluate_union_gap_at_time(
        ship,
        smokes,
        spatial_tolerance_m=1e-6,
        initial_polygon_sides=4,
        maximum_polygon_sides=4,
    )

    assert result.status is CertificationStatus.INDETERMINATE
