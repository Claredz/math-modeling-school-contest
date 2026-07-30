"""Tests for the isolated Q3 synthetic multiobjective toy problem."""

from __future__ import annotations

import random
from itertools import product
from time import perf_counter

import pytest

from experiments.toy_demos.common import ToyRunRecord
from experiments.toy_demos.q3_multiobjective import (
    assess_nsga2,
    enumerate_portfolios,
    exact_pareto_front,
    run_nsga2,
    solve_epsilon,
)

EXPECTED_FRONT = (
    ("DAD", 12, 4),
    ("DAA", 18, 5),
    ("AAA", 22, 6),
    ("ABD", 24, 7),
)


def _front_signature(front: tuple[object, ...]) -> tuple[tuple[str, int, int], ...]:
    return tuple((item.code, item.benefit, item.risk) for item in front)


def test_enumeration_has_all_sixty_four_legal_three_uav_portfolios() -> None:
    portfolios = enumerate_portfolios()

    assert len(portfolios) == 4**3 == 64
    assert len({item.code for item in portfolios}) == 64
    assert all(len(item.genes) == 3 for item in portfolios)
    assert all(gene in range(4) for item in portfolios for gene in item.genes)
    legal_codes = {"".join(chars) for chars in product("ABCD", repeat=3)}
    assert all(item.code in legal_codes for item in portfolios)


def test_exact_pareto_front_matches_the_hand_checked_manual_case() -> None:
    front = exact_pareto_front()

    assert _front_signature(front) == EXPECTED_FRONT
    assert all(
        not other.dominates(item)
        for item in front
        for other in enumerate_portfolios()
        if other != item
    )


@pytest.mark.parametrize(
    ("risk_limit", "expected"),
    [(6, ("AAA", 22, 6)), (5, ("DAA", 18, 5)), (4, ("DAD", 12, 4))],
)
def test_epsilon_constraint_selects_a_member_of_the_true_front(
    risk_limit: int,
    expected: tuple[str, int, int],
) -> None:
    result = solve_epsilon(risk_limit)

    assert result.feasible
    assert result.portfolio is not None
    assert (result.portfolio.code, result.portfolio.benefit, result.portfolio.risk) == expected
    assert result.portfolio in exact_pareto_front()
    assert isinstance(result.record, ToyRunRecord)
    assert result.record.passed_manual_case


def test_epsilon_limit_below_structural_minimum_is_reported_infeasible() -> None:
    result = solve_epsilon(3.99)

    assert not result.feasible
    assert result.portfolio is None
    assert result.failure_reason == "no portfolio satisfies the risk limit"
    assert not result.record.converged
    assert result.record.failure_reason == result.failure_reason


def test_nsga2_same_seed_is_reproducible_and_every_gene_is_legal() -> None:
    first = run_nsga2(seed=2026, population_size=40, generations=30)
    second = run_nsga2(seed=2026, population_size=40, generations=30)

    assert first.nondominated == second.nondominated
    assert first.coverage == second.coverage
    assert first.precision == second.precision
    assert first.dominated_count == second.dominated_count
    assert first.unique_evaluations == second.unique_evaluations
    assert all(gene in range(4) for item in first.nondominated for gene in item.genes)
    assert first.backend == "DEAP NSGA-II"


def test_nsga2_does_not_pollute_callers_python_random_state() -> None:
    random.seed(9182)
    state_before = random.getstate()

    run_nsga2(seed=17, population_size=40, generations=8)

    assert random.getstate() == state_before


def test_underbudgeted_nsga2_is_labeled_heuristic_incomplete_not_globally_converged() -> None:
    result = run_nsga2(seed=0, population_size=4, generations=1)

    assert result.coverage < 1.0
    assert not result.record.converged
    assert result.record.failure_reason == (
        "heuristic incomplete: not all exact Pareto points found"
    )
    assert result.dominated_count > 0


def test_multiseed_assessment_reports_coverage_precision_and_stability() -> None:
    assessment = assess_nsga2(
        seeds=(3, 7, 11, 19),
        population_size=40,
        generations=30,
    )

    assert len(assessment.runs) == 4
    assert 0.0 <= assessment.mean_coverage <= 1.0
    assert 0.0 <= assessment.minimum_coverage <= 1.0
    assert 0.0 <= assessment.mean_precision <= 1.0
    assert 0.0 <= assessment.mean_jaccard <= 1.0
    assert assessment.total_false_positives == sum(
        item.dominated_count for item in assessment.runs
    )
    assert assessment.backend == "DEAP NSGA-II"
    assert all(item.unique_evaluations <= 64 for item in assessment.runs)
    for run in assessment.runs:
        assert run.record.converged is (run.coverage == 1.0)
        assert run.record.failure_reason == (
            None
            if run.coverage == 1.0
            else "heuristic incomplete: not all exact Pareto points found"
        )


@pytest.mark.parametrize(
    ("function", "kwargs", "error"),
    [
        (run_nsga2, {"seed": True}, TypeError),
        (run_nsga2, {"seed": -1}, ValueError),
        (run_nsga2, {"seed": 1, "population_size": 6}, ValueError),
        (run_nsga2, {"seed": 1, "generations": 0}, ValueError),
        (solve_epsilon, {"risk_limit": float("nan")}, ValueError),
        (assess_nsga2, {"seeds": ()}, ValueError),
        (assess_nsga2, {"seeds": (1, 1)}, ValueError),
    ],
)
def test_public_entry_points_validate_inputs_strictly(
    function: object,
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        function(**kwargs)  # type: ignore[operator]


def test_complete_multiseed_demo_runs_under_thirty_seconds() -> None:
    started = perf_counter()
    assessment = assess_nsga2(
        seeds=(2, 5, 13, 23),
        population_size=40,
        generations=30,
    )
    elapsed = perf_counter() - started

    assert assessment.runs
    assert elapsed < 30.0
