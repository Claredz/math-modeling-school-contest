from __future__ import annotations

import numpy as np
import pytest

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
