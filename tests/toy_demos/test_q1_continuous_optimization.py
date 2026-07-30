import json
import math

import numpy as np
import pytest

from experiments.toy_demos.common import ToyRunRecord
from experiments.toy_demos.q1_continuous_optimization import (
    _RUNNERS,
    METHOD_STAGE_PATHS,
    METHODS,
    SURROGATE_DESCRIPTION,
    MethodResult,
    coverage,
    run_demo,
    run_method,
    verify_solution,
)


def test_manual_optimum_has_value_ten_and_is_independently_verified() -> None:
    assert coverage((0.0, 0.0)) == pytest.approx(10.0)

    verification = verify_solution((0.0, 0.0), reported_objective=10.0)

    assert verification.verified
    assert verification.feasible
    assert verification.gap == pytest.approx(0.0)


@pytest.mark.parametrize(
    "point",
    [
        (-1.4, 0.0),
        (1.4, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
    ],
)
def test_ellipse_boundary_is_feasible_but_strictly_suboptimal(
    point: tuple[float, float],
) -> None:
    verification = verify_solution(point, reported_objective=coverage(point))

    assert verification.feasible
    assert verification.objective < 10.0
    assert verification.gap > 0.0


def test_all_five_methods_respect_a_common_budget_and_pass_verification() -> None:
    results = run_demo(seed=20260731, budget=768)

    assert set(results) == set(METHODS) == {
        "multistart_slsqp",
        "de_slsqp",
        "pso_slsqp",
        "sobol_trust_constr",
        "shgo",
    }
    for method, result in results.items():
        assert isinstance(result, MethodResult), method
        assert result.method == method
        assert result.seed == 20260731
        assert result.budget == 768
        assert 0 < result.evaluation_count <= result.budget
        assert result.solver_success
        assert not result.budget_exhausted
        assert result.verified
        assert result.passed_manual_case
        assert tuple(result.stage_success) == METHOD_STAGE_PATHS[method]
        assert all(result.stage_success.values())
        assert result.failure is None
        assert result.objective == pytest.approx(10.0, abs=2e-5)
        assert result.gap <= 2e-5
        assert math.isfinite(result.runtime_s)
        assert 0.0 <= result.runtime_s < 30.0


@pytest.mark.parametrize("method", METHODS)
def test_each_method_is_reproducible_apart_from_runtime(method: str) -> None:
    first = run_method(method, seed=19, budget=768)
    second = run_method(method, seed=19, budget=768)

    assert first.as_dict(exclude_runtime=True) == second.as_dict(exclude_runtime=True)


@pytest.mark.parametrize("method", METHODS)
def test_tiny_budget_returns_a_structured_failure(method: str) -> None:
    result = run_method(method, seed=19, budget=1)

    assert result.method == method
    assert result.evaluation_count == 1
    assert result.budget == 1
    assert result.x != (0.0, 0.0)
    assert result.objective < 10.0
    assert not result.solver_success
    assert result.budget_exhausted
    assert not result.verified
    assert result.failure == "evaluation_budget_exhausted"


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        run_method("not-a-solver", seed=0, budget=768)


@pytest.mark.parametrize("seed", [True, -1, 1.5, float("nan"), "19"])
def test_seed_is_a_nonnegative_integer(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_method("pso_slsqp", seed=seed, budget=768)  # type: ignore[arg-type]


@pytest.mark.parametrize("budget", [True, False, 0, -1, 1.5, float("nan"), "768"])
def test_budget_is_a_positive_integer(budget: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_method("pso_slsqp", seed=19, budget=budget)  # type: ignore[arg-type]


def test_method_result_adapts_to_common_record_and_standard_json() -> None:
    result = run_method("sobol_trust_constr", seed=23, budget=768)

    record = result.to_toy_record()

    assert isinstance(record, ToyRunRecord)
    assert record.demo_name == "q1_continuous_optimization"
    assert record.solver == result.method
    assert record.seed == 23
    assert record.passed_manual_case
    assert record.metadata["stage_success"] == result.stage_success
    assert json.loads(record.to_json())["metadata"]["evaluation_count"] <= 768


def test_labels_are_bound_to_the_claimed_runner_and_stage_path() -> None:
    assert {method: runner.__name__ for method, runner in _RUNNERS.items()} == {
        "multistart_slsqp": "_multistart_slsqp",
        "de_slsqp": "_de_slsqp",
        "pso_slsqp": "_pso_slsqp",
        "sobol_trust_constr": "_sobol_trust_constr",
        "shgo": "_shgo",
    }
    assert METHOD_STAGE_PATHS == {
        "multistart_slsqp": ("sobol_multistart", "slsqp"),
        "de_slsqp": ("differential_evolution", "slsqp_polish"),
        "pso_slsqp": ("particle_swarm", "slsqp_polish"),
        "sobol_trust_constr": ("sobol_screen", "trust_constr", "slsqp_polish"),
        "shgo": ("shgo", "slsqp_polish"),
    }


def test_running_methods_does_not_mutate_numpy_global_rng() -> None:
    original_state = np.random.get_state()
    try:
        np.random.seed(314159)
        state_before = np.random.get_state()

        run_demo(seed=19, budget=768)

        state_after = np.random.get_state()
        assert state_before[0] == state_after[0]
        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2:] == state_after[2:]
    finally:
        np.random.set_state(original_state)


def test_surrogate_has_explicit_q1_coverage_meaning_and_nonformal_units() -> None:
    assert SURROGATE_DESCRIPTION == {
        "x": "normalized along-track burst-center offset",
        "y": "normalized cross-track burst-center offset",
        "objective": "dimensionless event-coverage utility surrogate",
        "upper_bound": "10 from nonnegative loss; not seconds and not a formal result",
    }
