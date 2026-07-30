import json

import numpy as np
import pytest

from experiments.toy_demos.common import ToyRunRecord, seeded_rng, timed_call


def make_record(**overrides: object) -> ToyRunRecord:
    values = {
        "demo_name": "symmetric-case",
        "solver": "analytic",
        "seed": 20260731,
        "objective": 1.25,
        "runtime_s": 0.01,
        "converged": True,
        "passed_manual_case": True,
        "failure_reason": None,
        "metadata": {"budget": 20, "method": "manual"},
    }
    values.update(overrides)
    return ToyRunRecord(**values)


@pytest.mark.parametrize("field", ["demo_name", "solver"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_record_rejects_empty_names(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_record(**{field: value})


@pytest.mark.parametrize("seed", [True, 1.5, "7", None])
def test_record_requires_an_integer_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_record(seed=seed)


@pytest.mark.parametrize("objective", [float("nan"), float("inf"), float("-inf")])
def test_record_rejects_nonfinite_objective(objective: float) -> None:
    with pytest.raises(ValueError):
        make_record(objective=objective)


@pytest.mark.parametrize("runtime_s", [-0.01, float("nan"), float("inf")])
def test_record_rejects_invalid_runtime(runtime_s: float) -> None:
    with pytest.raises(ValueError):
        make_record(runtime_s=runtime_s)


@pytest.mark.parametrize("field", ["converged", "passed_manual_case"])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_record_requires_boolean_status_fields(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        make_record(**{field: value})


def test_record_accepts_optional_failure_reason() -> None:
    record = make_record(
        converged=False,
        passed_manual_case=False,
        failure_reason="iteration limit",
    )

    assert record.failure_reason == "iteration limit"


def test_record_is_frozen() -> None:
    record = make_record()

    with pytest.raises((AttributeError, TypeError)):
        record.seed = 0  # type: ignore[misc]


def test_record_serializes_to_stable_standard_json() -> None:
    first = make_record(metadata={"z": 2, "a": {"y": 1, "x": 0}})
    second = make_record(metadata={"a": {"x": 0, "y": 1}, "z": 2})

    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()
    assert list(json.loads(first.to_json())) == [
        "converged",
        "demo_name",
        "failure_reason",
        "metadata",
        "objective",
        "passed_manual_case",
        "runtime_s",
        "seed",
        "solver",
    ]


def test_seeded_rng_repeats_the_same_sequence() -> None:
    first = seeded_rng(17).normal(size=4)
    second = seeded_rng(17).normal(size=4)

    np.testing.assert_array_equal(first, second)


def test_timed_call_returns_result_and_nonnegative_runtime() -> None:
    result, runtime_s = timed_call(lambda left, right: left + right, 2, 3)

    assert result == 5
    assert np.isfinite(runtime_s)
    assert runtime_s >= 0.0
