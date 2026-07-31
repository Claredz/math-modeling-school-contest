from __future__ import annotations

from smoke_defense.q1_rebuild import build_q1_problem
from smoke_defense.q3_rebuild import (
    construct_q3_plan,
    generate_q3_plan,
    verify_q3_plan,
)
from smoke_defense.scenario_matrix import generate_q1_rebuild_matrix


def test_q3_main_interpretation_has_exactly_three_bombs():
    problem = build_q1_problem(generate_q1_rebuild_matrix()[0])
    plan = construct_q3_plan(problem, burst_times_s=(8.0, 14.0, 20.0))
    assert len(plan.smokes) == 3
    assert len(plan.paths) == 3
    assert plan.interpretation == "exactly_one_bomb_per_uav"
    assert verify_q3_plan(problem, plan).status.value in {
        "certified_feasible",
        "certified_infeasible",
        "unresolved",
    }


def test_q3_generator_is_reproducible():
    problem = build_q1_problem(generate_q1_rebuild_matrix()[1])
    first, first_certificate = generate_q3_plan(problem)
    second, second_certificate = generate_q3_plan(problem)
    assert first.burst_times_s == second.burst_times_s
    assert first_certificate.joint.coverage_lower_s == second_certificate.joint.coverage_lower_s
