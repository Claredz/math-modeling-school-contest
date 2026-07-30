"""Tests for the isolated Q2 constraint-generation toy problem."""

from __future__ import annotations

import math

import pytest

from experiments.toy_demos.q2_constraint_generation import (
    CoverSolution,
    run_constraint_generation,
    separate_power_distance,
    solve_finite_master,
    verify_cover_independently,
)

INITIAL_WITNESSES = ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))


def test_initial_finite_master_misses_off_axis_points() -> None:
    solution = solve_finite_master(INITIAL_WITNESSES)

    assert solution.a == pytest.approx(0.5, abs=1e-8)
    assert solution.radius == pytest.approx(0.5, abs=1e-8)
    assert solution.objective == pytest.approx(0.25, abs=1e-8)


def test_separation_oracle_finds_unit_power_violation_at_pole() -> None:
    solution = CoverSolution(a=0.5, radius=0.5, objective=0.25)

    separated = separate_power_distance(solution)

    assert separated.witness[0] == pytest.approx(0.0, abs=1e-12)
    assert abs(separated.witness[1]) == pytest.approx(1.0, abs=1e-12)
    assert separated.violation == pytest.approx(1.0, abs=1e-12)
    assert separated.is_violated


def test_adding_oracle_constraint_converges_to_coincident_unit_disks() -> None:
    first = solve_finite_master(INITIAL_WITNESSES)
    witness = separate_power_distance(first).witness
    final = solve_finite_master((*INITIAL_WITNESSES, witness))

    assert final.a == pytest.approx(0.0, abs=1e-8)
    assert final.radius == pytest.approx(1.0, abs=1e-8)
    assert not separate_power_distance(final, tolerance=1e-10).is_violated


def test_constraint_generation_records_monotone_incumbent_upper_bounds() -> None:
    result = run_constraint_generation(seed=19, max_iterations=4)

    assert result.converged
    assert result.failure_reason is None
    assert result.solution.a == pytest.approx(0.0, abs=1e-8)
    assert result.solution.radius == pytest.approx(1.0, abs=1e-8)
    assert result.violation_upper_bounds[0] == pytest.approx(1.0)
    assert result.violation_upper_bounds[-1] == pytest.approx(0.0, abs=1e-10)
    assert all(
        later <= earlier
        for earlier, later in zip(
            result.violation_upper_bounds,
            result.violation_upper_bounds[1:],
            strict=False,
        )
    )


def test_zero_augmentation_budget_is_reported_as_unresolved() -> None:
    result = run_constraint_generation(seed=19, max_iterations=0)

    assert not result.converged
    assert result.failure_reason == "maximum constraint-generation iterations reached"
    assert result.solution.a == pytest.approx(0.5, abs=1e-8)
    assert result.violation_upper_bounds == pytest.approx((1.0,))
    assert result.added_witnesses == ()


def test_independent_verifier_does_not_accept_the_finite_grid_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.toy_demos.q2_constraint_generation as module

    def forbidden_oracle(*args: object, **kwargs: object) -> object:
        raise AssertionError("independent verifier reused the separation oracle")

    monkeypatch.setattr(module, "separate_power_distance", forbidden_oracle)

    missed = verify_cover_independently(CoverSolution(a=0.5, radius=0.5, objective=0.25))
    covered = verify_cover_independently(CoverSolution(a=0.0, radius=1.0, objective=1.0))

    assert not missed.passed
    assert missed.maximum_power_violation == pytest.approx(1.0)
    assert covered.passed
    assert covered.maximum_power_violation <= 1e-10
    assert math.isfinite(covered.maximum_power_violation)


def test_fixed_seed_produces_identical_result() -> None:
    first = run_constraint_generation(seed=2026, max_iterations=4)
    second = run_constraint_generation(seed=2026, max_iterations=4)

    assert first == second
