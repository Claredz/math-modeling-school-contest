"""Shared result contract and reproducibility helpers for toy demonstrations."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from time import perf_counter
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np
from numpy.random import Generator

ResultT = TypeVar("ResultT")


def _integer_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("seed must be nonnegative")
    return normalized


def _finite_number(value: object, field_name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if nonnegative and normalized < 0.0:
        raise ValueError(f"{field_name} must be nonnegative")
    return normalized


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError("metadata must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ToyRunRecord:
    """Immutable, machine-readable outcome of one isolated toy-demo run."""

    __hash__ = None

    demo_name: str
    solver: str
    seed: int
    objective: float
    runtime_s: float
    converged: bool
    passed_manual_case: bool
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("demo_name", "solver"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")

        object.__setattr__(self, "seed", _integer_seed(self.seed))
        object.__setattr__(self, "objective", _finite_number(self.objective, "objective"))
        object.__setattr__(
            self,
            "runtime_s",
            _finite_number(self.runtime_s, "runtime_s", nonnegative=True),
        )

        for field_name in ("converged", "passed_manual_case"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")

        if self.failure_reason is not None:
            if not isinstance(self.failure_reason, str):
                raise TypeError("failure_reason must be a string or None")
            if not self.failure_reason.strip():
                raise ValueError("failure_reason must not be empty")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""

        return {
            "demo_name": self.demo_name,
            "solver": self.solver,
            "seed": self.seed,
            "objective": self.objective,
            "runtime_s": self.runtime_s,
            "converged": self.converged,
            "passed_manual_case": self.passed_manual_case,
            "failure_reason": self.failure_reason,
            "metadata": _thaw_json(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize deterministically using sorted keys and strict JSON numbers."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def seeded_rng(seed: int) -> Generator:
    """Return an independent NumPy generator initialized from an integer seed."""

    return np.random.default_rng(_integer_seed(seed))


def timed_call(
    function: Callable[..., ResultT],
    /,
    *args: Any,
    **kwargs: Any,
) -> tuple[ResultT, float]:
    """Evaluate ``function`` and return its value with monotonic elapsed seconds."""

    started_at = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - started_at
