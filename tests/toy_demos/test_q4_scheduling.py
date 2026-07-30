"""Tests for the isolated Q4 synthetic online/offline scheduling toy."""

from __future__ import annotations

import random
from dataclasses import FrozenInstanceError, replace
from time import perf_counter

import pytest

import experiments.toy_demos.q4_scheduling as q4
from experiments.toy_demos.common import ToyRunRecord


def test_manual_instance_has_eighteen_combinations_and_unique_offline_optimum() -> None:
    batches = q4.default_batches()
    result = q4.enumerate_offline(batches, seed=2026)

    assert result.combinations_checked == 18
    assert result.selected_ids == ("T1-short", "T2-long")
    assert result.objective == 19
    assert result.converged and not result.unresolved
    assert result.verified
    assert isinstance(result.record, ToyRunRecord)
    assert result.record.passed_manual_case


def test_scipy_milp_matches_independent_enumeration_and_is_verified() -> None:
    batches = q4.default_batches()
    exact = q4.enumerate_offline(batches, seed=3)
    result = q4.solve_offline_milp(batches, seed=3)

    assert result.selected_ids == exact.selected_ids
    assert result.objective == exact.objective == 19
    assert result.converged and result.verified
    assert result.record.metadata["interpretation"] == "hindsight_upper_bound"


def test_rolling_zero_forecast_commits_whole_packages_and_scores_thirteen() -> None:
    result = q4.solve_rolling_zero_forecast(q4.default_batches(), seed=17)

    assert result.selected_ids == ("T1-long", "T3-only")
    assert result.objective == 13
    assert result.converged and result.verified
    assert tuple(trace.visible_threat_ids for trace in result.trace) == (
        ("T1",),
        ("T1", "T2"),
        ("T1", "T2", "T3"),
    )
    assert result.trace[0].selected_package_id == "T1-long"


def test_rolling_builds_a_fresh_milp_each_epoch_without_future_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_milp = q4.optimize.milp
    objective_vectors: list[tuple[float, ...]] = []

    def inspect_model(**kwargs: object) -> object:
        objective_vectors.append(tuple(float(item) for item in kwargs["c"]))  # type: ignore[index]
        return real_milp(**kwargs)

    monkeypatch.setattr(q4.optimize, "milp", inspect_model)

    result = q4.solve_rolling_zero_forecast(q4.default_batches(), seed=17)

    assert result.objective == 13
    assert objective_vectors == [
        (-8.0, -5.0),
        (-14.0, -8.0),
        (-14.0, -8.0, -5.0),
    ]


def test_rolling_skips_milp_but_records_trace_for_empty_release_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = (
        q4.ThreatBatch("T1", 0, (q4.TaskPackage("T1", "T1", (0,), 2),)),
        q4.ThreatBatch("T2", 2, (q4.TaskPackage("T2", "T2", (2,), 3),)),
    )
    real_milp = q4.optimize.milp
    calls = 0

    def count(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_milp(**kwargs)

    monkeypatch.setattr(q4.optimize, "milp", count)
    result = q4.solve_rolling_zero_forecast(batches)

    assert calls == 2
    assert tuple(item.time for item in result.trace) == (0, 1, 2)
    assert result.trace[1].selected_package_ids == ()


def test_causal_density_greedy_scores_eighteen_with_stable_tie_break() -> None:
    first = q4.solve_causal_greedy(q4.default_batches(), seed=41)
    second = q4.solve_causal_greedy(q4.default_batches(), seed=41)

    assert first.selected_ids == ("T1-short", "T2-short", "T3-only")
    assert first.objective == 18
    assert first == second
    assert first.record.metadata["rule"] == "value_density_then_value_then_id"


def test_greedy_accepts_multiple_nonconflicting_threats_released_same_epoch() -> None:
    batches = (
        q4.ThreatBatch("A", 0, (q4.TaskPackage("A", "A", (0,), 5),)),
        q4.ThreatBatch("B", 0, (q4.TaskPackage("B", "B", (1,), 4),)),
    )

    result = q4.solve_causal_greedy(batches)

    assert result.selected_ids == ("A", "B")
    assert result.trace[0].selected_package_ids == ("A", "B")
    assert result.objective == 9


def test_verifier_rejects_slot_conflicts_and_two_packages_for_one_threat() -> None:
    batches = q4.default_batches()

    conflict = q4.verify_selection(batches, ("T1-long", "T2-short"))
    duplicate = q4.verify_selection(batches, ("T1-long", "T1-short"))

    assert not conflict.valid
    assert conflict.failure == "slot capacity exceeded at slot 1"
    assert not duplicate.valid
    assert duplicate.failure == "more than one package selected for threat T1"


def test_t0_decision_is_isolated_from_same_history_different_future() -> None:
    baseline = q4.default_batches()
    modified = (
        baseline[0],
        replace(
            baseline[1],
            packages=tuple(replace(package, value=1000) for package in baseline[1].packages),
        ),
        replace(
            baseline[2],
            packages=tuple(replace(package, value=2000) for package in baseline[2].packages),
        ),
    )

    first = q4.solve_rolling_zero_forecast(baseline, seed=8)
    second = q4.solve_rolling_zero_forecast(modified, seed=8)

    assert first.trace[0].visible_threat_ids == second.trace[0].visible_threat_ids == ("T1",)
    assert first.trace[0].selected_package_id == second.trace[0].selected_package_id


def test_milp_failure_is_structured_and_never_silently_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Failed:
        success = False
        message = "synthetic solver failure"
        x = None

    monkeypatch.setattr(q4.optimize, "milp", lambda **_: Failed())

    result = q4.solve_offline_milp(q4.default_batches(), seed=5)

    assert not result.converged
    assert result.unresolved
    assert not result.verified
    assert result.selected_ids == ()
    assert result.failure == "MILP failed: synthetic solver failure"
    assert result.record.failure_reason == result.failure


@pytest.mark.parametrize("solver", [q4.solve_offline_milp, q4.solve_rolling_zero_forecast])
def test_milp_exception_is_structured_and_not_propagated(
    monkeypatch: pytest.MonkeyPatch,
    solver: object,
) -> None:
    def explode(**_: object) -> object:
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(q4.optimize, "milp", explode)

    result = solver(q4.default_batches(), seed=5)  # type: ignore[operator]

    assert not result.converged
    assert result.unresolved
    assert not result.verified
    assert result.failure == "milp_exception:RuntimeError"
    assert result.record.failure_reason == result.failure


def test_milp_accepts_a_different_selection_with_the_same_optimal_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = (
        q4.ThreatBatch(
            "T1",
            0,
            (
                q4.TaskPackage("T1-a", "T1", (0,), 5),
                q4.TaskPackage("T1-b", "T1", (1,), 5),
            ),
        ),
    )

    class AlternateOptimum:
        success = True
        message = "ok"
        x = (0.0, 1.0)

    monkeypatch.setattr(q4.optimize, "milp", lambda **_: AlternateOptimum())

    result = q4.solve_offline_milp(batches, seed=5)

    assert result.converged and result.verified
    assert result.selected_ids == ("T1-b",)
    assert result.objective == 5


@pytest.mark.parametrize(
    "bad_x",
    [
        (1.0,),
        (float("nan"),) * 5,
        (0.5,) + (0.0,) * 4,
        (1.2,) + (0.0,) * 4,
        (1.0, 0.0, 1.0, 0.0, 0.0),
    ],
)
def test_offline_rejects_malformed_or_constraint_violating_milp_vectors(
    monkeypatch: pytest.MonkeyPatch,
    bad_x: tuple[float, ...],
) -> None:
    class Invalid:
        success = True
        message = "claimed success"
        x = bad_x

    monkeypatch.setattr(q4.optimize, "milp", lambda **_: Invalid())

    with pytest.raises(RuntimeError, match="invalid MILP solution"):
        q4.solve_offline_milp(q4.default_batches())


def test_rolling_rejects_solver_selection_for_a_zero_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_milp = q4.optimize.milp
    call = 0

    def violate_bound(**kwargs: object) -> object:
        nonlocal call
        call += 1
        result = real_milp(**kwargs)
        if call == 2:
            result.x[:] = 0
            result.x[0] = 1
        return result

    monkeypatch.setattr(q4.optimize, "milp", violate_bound)
    result = q4.solve_rolling_zero_forecast(q4.default_batches())

    assert not result.converged and result.unresolved
    assert result.failure == "invalid_milp_solution:variable outside bounds"


def test_milp_enumeration_mismatch_is_a_hard_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_milp = q4.optimize.milp

    def wrong_solution(**kwargs: object) -> object:
        result = real_milp(**kwargs)
        result.x[:] = 0
        result.x[0] = 1
        return result

    monkeypatch.setattr(q4.optimize, "milp", wrong_solution)

    with pytest.raises(RuntimeError, match="MILP/enumeration mismatch"):
        q4.solve_offline_milp(q4.default_batches(), seed=5)


def test_inputs_are_strict_immutable_and_seed_does_not_pollute_random_state() -> None:
    package = q4.TaskPackage("X-a", "X", (0,), 1)
    with pytest.raises(FrozenInstanceError):
        package.value = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        q4.TaskPackage("bad", "X", (0,), True)
    with pytest.raises(ValueError):
        q4.ThreatBatch("X", -1, (package,))
    with pytest.raises(ValueError):
        q4.enumerate_offline(q4.default_batches(), seed=-1)

    random.seed(9182)
    state_before = random.getstate()
    q4.solve_causal_greedy(q4.default_batches(), seed=6)
    assert random.getstate() == state_before


def test_result_contracts_deep_freeze_and_reject_inconsistent_status() -> None:
    record = q4.enumerate_offline(q4.default_batches()).record
    result = q4.ScheduleResult(
        ["T1-short"],  # type: ignore[arg-type]
        5,
        True,
        False,
        True,
        None,
        replace(record, objective=5),
        trace=[q4.DecisionTrace(0, ("T1",), ("T1-short",), (0,))],  # type: ignore[arg-type]
    )

    assert result.selected_ids == ("T1-short",)
    assert isinstance(result.trace, tuple)
    with pytest.raises(ValueError):
        q4.ScheduleResult((), 0, True, True, True, None, record)
    with pytest.raises(ValueError):
        q4.VerificationResult(True, float("nan"), None)


def test_result_contract_rejects_duplicate_negative_and_ambiguous_statuses() -> None:
    record = q4.enumerate_offline(q4.default_batches()).record
    with pytest.raises(ValueError, match="selected_ids must be unique"):
        q4.ScheduleResult(("x", "x"), 19, True, False, True, None, record)
    with pytest.raises(ValueError, match="objective must be nonnegative"):
        q4.ScheduleResult((), -1, True, False, True, None, replace(record, objective=-1))
    with pytest.raises(ValueError, match="failure must not be empty"):
        q4.ScheduleResult(
            (),
            0,
            False,
            True,
            False,
            " ",
            replace(
                record,
                objective=0,
                converged=False,
                passed_manual_case=False,
                failure_reason="failed",
            ),
        )
    with pytest.raises(ValueError, match="must be either successful or unresolved"):
        q4.ScheduleResult(
            (),
            0,
            False,
            False,
            False,
            None,
            replace(record, objective=0, converged=False, passed_manual_case=False),
        )
    with pytest.raises(ValueError, match="unresolved result cannot be verified"):
        q4.ScheduleResult(
            (),
            0,
            False,
            True,
            True,
            "failed",
            replace(
                record,
                objective=0,
                converged=False,
                passed_manual_case=True,
                failure_reason="failed",
            ),
        )
    with pytest.raises(ValueError, match="objective must be nonnegative"):
        q4.VerificationResult(False, -1, "bad")


@pytest.mark.parametrize(
    "record_patch",
    [
        {"objective": 18},
        {"converged": False},
        {"failure_reason": "unexpected"},
        {"passed_manual_case": False},
    ],
)
def test_schedule_result_must_match_its_toy_run_record(
    record_patch: dict[str, object],
) -> None:
    record = q4.enumerate_offline(q4.default_batches()).record

    with pytest.raises(ValueError, match="record must match schedule result"):
        q4.ScheduleResult(
            ("T1-short", "T2-long"),
            19,
            True,
            False,
            True,
            None,
            replace(record, **record_patch),
        )


def test_demo_is_isolated_and_runs_under_thirty_seconds() -> None:
    started = perf_counter()
    result = q4.run_demo(seed=2026)

    assert perf_counter() - started < 30
    assert result["offline_milp"].objective >= result["greedy"].objective
    assert result["offline_milp"].objective >= result["rolling"].objective
    assert all(
        set(item.selected_ids)
        <= {"T1-long", "T1-short", "T2-long", "T2-short", "T3-only"}
        for item in result.values()
    )
