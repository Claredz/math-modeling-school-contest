"""Tests for the isolated Q2 joint discrete--continuous toy problem."""

from __future__ import annotations

import math
from time import perf_counter
from types import SimpleNamespace

import pytest

import experiments.toy_demos.q2_joint_prototype as joint_module
from experiments.toy_demos.common import ToyRunRecord
from experiments.toy_demos.q2_joint_prototype import (
    BOMB_PARAMETERS,
    GRID_BOUND_SOURCE,
    BombSchedule,
    GridBounds,
    enumerate_grid_bounds,
    evaluate_coverage,
    run_q2_joint_demo,
    sample_objective,
    solve_candidate_polish,
    solve_separation_oracle,
    verify_schedule_exactly,
)


def test_grid_enumeration_returns_a_manual_feasible_lower_bound_and_legal_upper_bound() -> None:
    bounds = enumerate_grid_bounds(grid_step=0.25)

    checked = verify_schedule_exactly(bounds.schedule)
    assert bounds.schedule == BombSchedule(("B", "C", "A"), (0.5, 2.0, 3.5))
    assert bounds.lower_bound == pytest.approx(5.0 / 11.0, abs=1e-12)
    assert bounds.lower_bound == pytest.approx(checked.objective, abs=1e-12)
    assert bounds.lower_bound <= bounds.global_upper_bound
    assert bounds.global_upper_bound - bounds.lower_bound == pytest.approx(
        bounds.lipschitz_constant * bounds.grid_step,
        abs=1e-12,
    )
    assert bounds.evaluated_schedules > 0
    assert len(bounds.schedule.bomb_types) <= 3


def test_discrete_and_continuous_schedule_decisions_are_explicit_and_feasible() -> None:
    schedule = BombSchedule(("B", "A", "C"), (0.5, 1.25, 3.75))

    checked = verify_schedule_exactly(schedule)

    assert schedule.bomb_types == ("B", "A", "C")
    assert schedule.burst_times == pytest.approx((0.5, 1.25, 3.75))
    assert all(
        later - earlier >= 0.5
        for earlier, later in zip(
            schedule.burst_times, schedule.burst_times[1:], strict=False
        )
    )
    assert math.isfinite(checked.objective)
    assert checked.objective >= 0.0
    assert evaluate_coverage(schedule, checked.worst_time) >= 0.0


def test_both_routes_are_independently_verified_and_bracketed_by_global_bounds() -> None:
    bounds = enumerate_grid_bounds(grid_step=0.25)
    candidate = solve_candidate_polish(seed=23, global_bounds=bounds)
    oracle = solve_separation_oracle(seed=23, global_bounds=bounds, max_iterations=8)

    for result in (candidate, oracle):
        independently_checked = verify_schedule_exactly(result.schedule)
        assert result.verified_objective == pytest.approx(
            independently_checked.objective, abs=1e-9
        )
        assert result.global_lower_bound <= result.verified_objective + 1e-10
        assert result.verified_objective <= result.global_upper_bound + 1e-10
        assert result.global_gap == pytest.approx(
            result.global_upper_bound - result.global_lower_bound
        )
        assert result.master_values
        assert isinstance(result.record, ToyRunRecord)
        assert result.record.metadata["bound_source"] == "Lipschitz grid covering bound"
        assert not result.globally_resolved
        assert result.unresolved
        assert not result.converged
        assert result.record.metadata["local_solver_converged"] == (
            result.local_solver_converged
        )


def test_zero_oracle_augmentation_budget_stays_unresolved_and_exposes_gap() -> None:
    bounds = enumerate_grid_bounds(grid_step=0.25)
    result = solve_separation_oracle(seed=7, global_bounds=bounds, max_iterations=0)

    assert not result.converged
    assert result.failure_reason == "maximum separation-oracle iterations reached"
    assert result.global_gap > 0.0
    assert result.record.failure_reason == result.failure_reason
    assert result.record.metadata["unresolved"]
    assert result.record.metadata["global_gap"] == pytest.approx(result.global_gap)
    assert result.record.metadata["master_is_global_upper_bound"] is False


def test_routes_match_or_improve_the_quarter_grid_quality() -> None:
    bounds = enumerate_grid_bounds(grid_step=0.25)
    candidate = solve_candidate_polish(seed=41, global_bounds=bounds)
    oracle = solve_separation_oracle(seed=41, global_bounds=bounds, max_iterations=8)

    assert candidate.verified_objective >= bounds.lower_bound - 1e-6
    assert oracle.verified_objective >= bounds.lower_bound - 2e-3


def test_exact_breakpoint_verifier_finds_a_dip_hidden_by_sparse_samples() -> None:
    schedule = BombSchedule(("B", "A", "C"), (0.60, 2.53, 3.48))

    sparse = sample_objective(schedule, sample_times=(0.0, 1.0, 2.0, 3.0, 4.0))
    exact = verify_schedule_exactly(schedule)

    assert sparse > exact.objective + 0.05
    assert exact.worst_time not in (0.0, 1.0, 2.0, 3.0, 4.0)


def test_fixed_seed_produces_identical_decisions_and_auditable_run_record() -> None:
    first = run_q2_joint_demo(seed=2026, max_iterations=8)
    second = run_q2_joint_demo(seed=2026, max_iterations=8)

    assert first.grid_bounds == second.grid_bounds
    assert first.candidate_route.schedule == second.candidate_route.schedule
    assert first.oracle_route.schedule == second.oracle_route.schedule
    assert first.candidate_route.verified_objective == pytest.approx(
        second.candidate_route.verified_objective, abs=1e-12
    )
    assert first.oracle_route.verified_objective == pytest.approx(
        second.oracle_route.verified_objective, abs=1e-12
    )
    assert first.candidate_route.record.seed == 2026
    assert first.oracle_route.record.seed == 2026


@pytest.mark.parametrize(
    ("callable_name", "kwargs", "error"),
    [
        ("enumerate", {"grid_step": 0.3}, ValueError),
        ("candidate", {"seed": -1}, ValueError),
        ("oracle", {"seed": True}, TypeError),
        ("oracle", {"max_iterations": -1}, ValueError),
    ],
)
def test_public_solvers_reject_inputs_that_break_the_bound_certificate(
    callable_name: str,
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    if callable_name == "enumerate":
        with pytest.raises(error):
            enumerate_grid_bounds(**kwargs)
        return

    bounds = enumerate_grid_bounds(grid_step=0.25)
    solver = solve_candidate_polish if callable_name == "candidate" else solve_separation_oracle
    with pytest.raises(error):
        solver(global_bounds=bounds, **kwargs)


@pytest.mark.parametrize(
    ("bomb_types", "burst_times"),
    [
        (("A", "B"), (1.0, 1.49)),
        (("A", "A"), (0.0, 1.0)),
        (("Z",), (1.0,)),
        (("A",), (4.1,)),
    ],
)
def test_schedule_validation_is_strict(
    bomb_types: tuple[str, ...], burst_times: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        BombSchedule(bomb_types, burst_times)


def test_schedule_normalises_sequences_and_bomb_catalogue_is_read_only() -> None:
    schedule = BombSchedule(["A", "C"], [0, 2])  # type: ignore[arg-type]

    assert schedule.bomb_types == ("A", "C")
    assert schedule.burst_times == (0.0, 2.0)
    with pytest.raises(TypeError):
        BOMB_PARAMETERS["A"] = (9.0, 9.0)  # type: ignore[index]


def test_grid_bounds_reject_forged_values_and_provenance() -> None:
    valid = enumerate_grid_bounds(grid_step=0.25)
    shared = {
        "schedule": valid.schedule,
        "lower_bound": valid.lower_bound,
        "global_upper_bound": valid.global_upper_bound,
        "grid_step": valid.grid_step,
        "lipschitz_constant": valid.lipschitz_constant,
        "evaluated_schedules": valid.evaluated_schedules,
        "bound_source": GRID_BOUND_SOURCE,
    }

    for changes in (
        {"lower_bound": valid.lower_bound + 0.1},
        {"global_upper_bound": valid.lower_bound - 0.1},
        {"grid_step": math.nan},
        {"lipschitz_constant": -1.0},
        {"evaluated_schedules": 0},
        {"bound_source": "local master guess"},
    ):
        with pytest.raises((TypeError, ValueError)):
            GridBounds(**(shared | changes))


def test_solvers_recompute_grid_optimum_and_reject_nonoptimal_certificate() -> None:
    legitimate = enumerate_grid_bounds(grid_step=0.25)
    nonoptimal_schedule = BombSchedule(("A",), (0.0,))
    nonoptimal_lower = verify_schedule_exactly(nonoptimal_schedule).objective
    fine_step = 0.0625
    forged = GridBounds(
        schedule=nonoptimal_schedule,
        lower_bound=nonoptimal_lower,
        global_upper_bound=(
            nonoptimal_lower + legitimate.lipschitz_constant * fine_step
        ),
        grid_step=fine_step,
        lipschitz_constant=legitimate.lipschitz_constant,
        evaluated_schedules=1,
        bound_source=GRID_BOUND_SOURCE,
    )

    assert forged.global_upper_bound < legitimate.lower_bound
    with pytest.raises(ValueError, match="recomputed exhaustive grid optimum"):
        solve_candidate_polish(seed=3, global_bounds=forged)
    with pytest.raises(ValueError, match="recomputed exhaustive grid optimum"):
        solve_separation_oracle(seed=3, global_bounds=forged, max_iterations=0)


def test_minimize_failure_is_propagated_and_never_labelled_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_minimize(
        function: object, x0: object, *args: object, **kwargs: object
    ) -> SimpleNamespace:
        del function, args, kwargs
        return SimpleNamespace(
            x=x0,
            success=False,
            status=9,
            message="forced iteration limit",
        )

    monkeypatch.setattr(joint_module, "minimize", failed_minimize)
    bounds = enumerate_grid_bounds(grid_step=0.25)
    candidate = solve_candidate_polish(seed=3, global_bounds=bounds)
    oracle = solve_separation_oracle(seed=3, global_bounds=bounds, max_iterations=2)

    for result in (candidate, oracle):
        assert not result.local_solver_converged
        assert not result.globally_resolved
        assert result.unresolved
        assert not result.converged
        assert result.failure_reason is not None
        assert "status=9" in result.failure_reason
        assert "forced iteration limit" in result.failure_reason


@pytest.mark.parametrize("tolerance", [True, 0.0, -1.0, math.nan, math.inf])
def test_oracle_rejects_invalid_tolerance(tolerance: object) -> None:
    bounds = enumerate_grid_bounds(grid_step=0.25)
    with pytest.raises((TypeError, ValueError)):
        solve_separation_oracle(global_bounds=bounds, tolerance=tolerance)  # type: ignore[arg-type]


@pytest.mark.parametrize("samples", [(True,), ("1.0",), (math.nan,), ()])
def test_finite_sampler_rejects_nonreal_or_invalid_times(samples: tuple[object, ...]) -> None:
    schedule = BombSchedule(("A",), (1.0,))
    with pytest.raises((TypeError, ValueError)):
        sample_objective(schedule, sample_times=samples)  # type: ignore[arg-type]


def test_repeated_separator_witness_has_a_distinct_stagnation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_verify = joint_module.verify_schedule_exactly

    def repeated_witness(schedule: BombSchedule) -> object:
        checked = original_verify(schedule)
        return joint_module.ExactVerification(
            objective=checked.objective,
            worst_time=0.0,
            breakpoints=checked.breakpoints,
        )

    monkeypatch.setattr(joint_module, "verify_schedule_exactly", repeated_witness)
    bounds = enumerate_grid_bounds(grid_step=0.25)
    result = solve_separation_oracle(seed=8, global_bounds=bounds, max_iterations=4)

    assert result.unresolved
    assert result.failure_reason == "separation oracle stagnated on a repeated witness"


def test_complete_demo_records_runtime_without_a_machine_specific_tight_limit() -> None:
    started = perf_counter()
    result = run_q2_joint_demo(seed=5, max_iterations=8)
    elapsed = perf_counter() - started

    assert result.candidate_route.verified_objective > 0.0
    assert result.oracle_route.verified_objective > 0.0
    assert math.isfinite(elapsed)
    assert elapsed < 120.0
