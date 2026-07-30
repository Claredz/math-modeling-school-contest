import math

import pytest

from experiments.toy_demos.q1_continuous_optimization import (
    METHODS,
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
        assert result.budget == 768
        assert 0 < result.evaluation_count <= result.budget
        assert result.solver_success
        assert not result.budget_exhausted
        assert result.verified
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
