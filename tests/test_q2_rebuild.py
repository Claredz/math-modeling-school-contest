from __future__ import annotations

import numpy as np
import pytest

from smoke_defense.coverage import CertificationStatus, CoverageCertificate
from smoke_defense.dynamics import ShipMotion
from smoke_defense.events import ClosedInterval
from smoke_defense.q1_rebuild import build_q1_problem
from smoke_defense.q2_rebuild import (
    Q2CertificationStatus,
    certify_joint_coverage,
    construct_q2_plan,
    solve_q2_candidates,
    verify_q2_plan,
)
from smoke_defense.scenario_matrix import generate_q1_rebuild_matrix
from smoke_defense.smoke import SmokeCloud


def _time_window_verifier(monkeypatch, covered_windows):
    def fake_certify(target, covers, *, initial_polygon_sides, maximum_polygon_sides):
        center_time = float(target.center_m[0])
        half_width = max(0.0, float(target.radius_m) - 1.0)
        contains = any(
            left <= center_time - half_width + 1e-12
            and center_time + half_width <= right + 1e-12
            for left, right in covered_windows
        )
        exact_inside = any(left <= center_time <= right for left, right in covered_windows)
        if contains or (half_width <= 1e-12 and exact_inside):
            return CoverageCertificate(CertificationStatus.CERTIFIED_FEASIBLE)
        return CoverageCertificate(
            CertificationStatus.CERTIFIED_INFEASIBLE,
            witness_m=np.asarray(target.center_m, dtype=float),
            reason="synthetic exposed witness",
        )

    monkeypatch.setattr("smoke_defense.q2_rebuild.certify_union_coverage", fake_certify)


def test_joint_verifier_accepts_spatially_complementary_smokes():
    ship = ShipMotion((0.0, 0.0), 0.0, 0.0)
    smokes = (
        SmokeCloud(0.0, np.array([-55.0, 0.0]), maximum_radius_m=120.0),
        SmokeCloud(0.0, np.array([55.0, 0.0]), maximum_radius_m=120.0),
    )

    result = certify_joint_coverage(
        ship_position=ship.position,
        detection_components=(ClosedInterval(0.0, 1.0),),
        smokes=smokes,
        ship_radius_m=80.0,
        ship_speed_bound_mps=0.0,
    )

    assert result.status is Q2CertificationStatus.CERTIFIED_FEASIBLE
    assert result.coverage_lower_s == pytest.approx(1.0)
    assert result.coverage_upper_s == pytest.approx(1.0)


def test_joint_verifier_preserves_unresolved_at_polygon_tolerance():
    ship = ShipMotion((0.0, 0.0), 0.0, 0.0)
    smoke = SmokeCloud(0.0, np.zeros(2), maximum_radius_m=80.0)

    result = certify_joint_coverage(
        ship_position=ship.position,
        detection_components=(ClosedInterval(0.0, 1.0),),
        smokes=(smoke,),
        ship_radius_m=80.0,
        ship_speed_bound_mps=0.0,
        initial_polygon_sides=8,
        maximum_polygon_sides=8,
        time_tolerance_s=0.5,
    )

    assert result.status is Q2CertificationStatus.UNRESOLVED
    assert result.unresolved_intervals
    assert not result.certified_exposed_intervals
    assert result.total_exposure_lower_s == pytest.approx(0.0)
    assert result.maximum_continuous_exposure_s == pytest.approx(0.0)


def test_q2_plan_enforces_delays_and_one_second_drop_spacing():
    scenario = next(
        item
        for item in generate_q1_rebuild_matrix()
        if item.scenario_id == "q1_rebuild_front_d10000"
    )
    problem = build_q1_problem(scenario)
    plan = construct_q2_plan(problem, burst_times_s=(12.0, 18.0))

    assert len(plan.bombs) == 2
    assert plan.bombs[1].drop_time_s - plan.bombs[0].drop_time_s >= 1.0
    assert all(b.drop_time_s - b.command_time_s == pytest.approx(2.0) for b in plan.bombs)
    assert all(b.burst_time_s - b.drop_time_s == pytest.approx(3.5) for b in plan.bombs)
    assert verify_q2_plan(problem, plan).status in {
        "certified_feasible",
        "certified_infeasible",
        "unresolved",
    }


def test_q2_candidate_search_keeps_multi_bomb_warm_start_and_center_decisions():
    scenario = generate_q1_rebuild_matrix()[0]
    problem = build_q1_problem(scenario)
    result = solve_q2_candidates(
        problem,
        warm_burst_times_s=(5.5,),
        warm_center_times_s=(12.0,),
        maximum_candidates=12,
        polish=False,
    )
    assert result.candidates
    assert any(len(item.burst_times_s) >= 2 for item in result.candidates)
    assert len(result.best.center_times_s) == len(result.best.burst_times_s)
    assert result.best.certificate.status in {
        Q2CertificationStatus.CERTIFIED_FEASIBLE,
        Q2CertificationStatus.CERTIFIED_INFEASIBLE,
        Q2CertificationStatus.UNRESOLVED,
    }


def test_q2_verifier_reports_longest_exposed_interval_not_total_exposure(monkeypatch):
    _time_window_verifier(monkeypatch, ((0.0, 2.0), (4.0, 6.0)))

    result = certify_joint_coverage(
        ship_position=lambda time_s: np.array([time_s, 0.0]),
        detection_components=(ClosedInterval(0.0, 8.0),),
        smokes=(),
        ship_radius_m=1.0,
        ship_speed_bound_mps=1.0,
        time_tolerance_s=1e-3,
    )

    assert result.status is Q2CertificationStatus.CERTIFIED_INFEASIBLE
    assert result.total_exposure_lower_s == pytest.approx(4.0, abs=2e-3)
    assert result.maximum_continuous_exposure_s == pytest.approx(2.0, abs=2e-3)
    assert result.maximum_exposure_lower_s == pytest.approx(2.0, abs=2e-3)
    assert result.maximum_exposure_upper_s == pytest.approx(2.0, abs=2e-3)
    assert len(result.certified_exposed_intervals) == 2
    assert len(result.certified_covered_intervals) == 2


def test_q2_verifier_merges_adjacent_certified_exposure_intervals(monkeypatch):
    _time_window_verifier(monkeypatch, ((0.0, 1.0),))

    result = certify_joint_coverage(
        ship_position=lambda time_s: np.array([time_s, 0.0]),
        detection_components=(ClosedInterval(0.0, 3.0),),
        smokes=(),
        ship_radius_m=1.0,
        ship_speed_bound_mps=1.0,
        time_tolerance_s=1e-3,
    )

    assert len(result.certified_exposed_intervals) == 1
    exposed = result.certified_exposed_intervals[0]
    assert exposed.start_s < 1.0
    assert exposed.end_s == pytest.approx(3.0, abs=2e-3)
    assert result.maximum_continuous_exposure_s == pytest.approx(
        exposed.duration_s, abs=2e-3
    )


def test_q2_joint_gain_is_increment_over_best_single_smoke_baseline(monkeypatch):
    _time_window_verifier(monkeypatch, ((0.0, 3.0),))
    smokes = (
        SmokeCloud(0.0, np.array([100.0, 100.0]), maximum_radius_m=1.0),
        SmokeCloud(0.0, np.array([120.0, 120.0]), maximum_radius_m=1.0),
    )

    result = certify_joint_coverage(
        ship_position=lambda time_s: np.array([time_s, 0.0]),
        detection_components=(ClosedInterval(0.0, 4.0),),
        smokes=smokes,
        ship_radius_m=1.0,
        ship_speed_bound_mps=1.0,
        time_tolerance_s=1e-3,
    )

    assert result.best_single_smoke_coverage_lower_s == pytest.approx(3.0, abs=2e-3)
    assert result.coverage_lower_s == pytest.approx(3.0, abs=2e-3)
    assert result.joint_gain_s == pytest.approx(0.0, abs=2e-3)
    assert result.best_single_smoke_candidate_id == "smoke_0"


def test_q2_verifier_exposes_interval_merge_helper_contract():
    from smoke_defense.q2_rebuild import merge_certified_intervals

    merged = merge_certified_intervals(
        (ClosedInterval(1.0, 2.0), ClosedInterval(2.0, 3.0), ClosedInterval(3.0, 4.0))
    )

    assert merged == (ClosedInterval(1.0, 4.0),)
