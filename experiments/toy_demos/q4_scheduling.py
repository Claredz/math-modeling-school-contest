"""Synthetic Q4 comparison of hindsight and causal scheduling policies.

The instance is deliberately tiny.  Offline enumeration is an independent
audit oracle; the rolling and greedy policies only inspect released threats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np
from scipy import optimize

from experiments.toy_demos.common import ToyRunRecord


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _integer(value: object, field_name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    normalized = int(value)
    if nonnegative and normalized < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return normalized


def _seed(value: object) -> int:
    return _integer(value, "seed")


def _value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("value must be finite and positive")
    return normalized


@dataclass(frozen=True, slots=True)
class TaskPackage:
    """One indivisible scheduling option for one threat."""

    package_id: str
    threat_id: str
    slots: tuple[int, ...]
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_id", _nonempty(self.package_id, "package_id"))
        object.__setattr__(self, "threat_id", _nonempty(self.threat_id, "threat_id"))
        if not isinstance(self.slots, tuple):
            raise TypeError("slots must be a tuple")
        normalized_slots = tuple(_integer(slot, "slot") for slot in self.slots)
        if not normalized_slots:
            raise ValueError("slots must not be empty")
        if len(set(normalized_slots)) != len(normalized_slots):
            raise ValueError("slots must be unique")
        object.__setattr__(self, "slots", normalized_slots)
        object.__setattr__(self, "value", _value(self.value))


@dataclass(frozen=True, slots=True)
class ThreatBatch:
    """Threat and all task packages revealed at one release time."""

    threat_id: str
    release_time: int
    packages: tuple[TaskPackage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "threat_id", _nonempty(self.threat_id, "threat_id"))
        object.__setattr__(self, "release_time", _integer(self.release_time, "release_time"))
        if not isinstance(self.packages, tuple):
            raise TypeError("packages must be a tuple")
        if not self.packages:
            raise ValueError("packages must not be empty")
        if any(not isinstance(package, TaskPackage) for package in self.packages):
            raise TypeError("packages must contain TaskPackage values")
        if any(package.threat_id != self.threat_id for package in self.packages):
            raise ValueError("package threat_id must match its batch")
        if any(min(package.slots) < self.release_time for package in self.packages):
            raise ValueError("package cannot occupy a slot before release")
        if len({package.package_id for package in self.packages}) != len(self.packages):
            raise ValueError("package ids must be unique within a batch")


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Auditable information available to a causal policy at one time."""

    time: int
    visible_threat_ids: tuple[str, ...]
    selected_package_ids: tuple[str, ...]
    committed_slots: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _integer(self.time, "time"))
        if not isinstance(self.visible_threat_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.visible_threat_ids
        ):
            raise TypeError("visible_threat_ids must be a tuple of nonempty strings")
        if not isinstance(self.selected_package_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.selected_package_ids
        ):
            raise TypeError("selected_package_ids must be a tuple of nonempty strings")
        if not isinstance(self.committed_slots, tuple):
            raise TypeError("committed_slots must be a tuple")
        object.__setattr__(
            self,
            "committed_slots",
            tuple(_integer(slot, "committed slot") for slot in self.committed_slots),
        )

    @property
    def selected_package_id(self) -> str | None:
        """Compatibility view for earlier single-selection traces."""

        return self.selected_package_ids[0] if self.selected_package_ids else None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    objective: float
    failure: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be boolean")
        if isinstance(self.objective, bool) or not isinstance(self.objective, Real):
            raise TypeError("objective must be real")
        normalized = float(self.objective)
        if not math.isfinite(normalized):
            raise ValueError("objective must be finite")
        object.__setattr__(self, "objective", normalized)
        if self.valid is (self.failure is not None):
            raise ValueError("valid and failure must be consistent")


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """Immutable outcome shared by exact, MILP, and causal solvers."""

    selected_ids: tuple[str, ...]
    objective: float
    converged: bool
    unresolved: bool
    verified: bool
    failure: str | None
    record: ToyRunRecord = field(compare=False, repr=False)
    trace: tuple[DecisionTrace, ...] = ()
    combinations_checked: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.selected_ids, (tuple, list)) or any(
            not isinstance(item, str) or not item.strip() for item in self.selected_ids
        ):
            raise TypeError("selected_ids must contain nonempty strings")
        object.__setattr__(self, "selected_ids", tuple(self.selected_ids))
        if isinstance(self.objective, bool) or not isinstance(self.objective, Real):
            raise TypeError("objective must be real")
        objective = float(self.objective)
        if not math.isfinite(objective):
            raise ValueError("objective must be finite")
        object.__setattr__(self, "objective", objective)
        for name in ("converged", "unresolved", "verified"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.converged and self.unresolved:
            raise ValueError("converged and unresolved cannot both be true")
        if self.converged and self.failure is not None:
            raise ValueError("a converged result cannot have a failure")
        if self.unresolved and self.failure is None:
            raise ValueError("an unresolved result must have a failure")
        if not isinstance(self.trace, (tuple, list)) or any(
            not isinstance(item, DecisionTrace) for item in self.trace
        ):
            raise TypeError("trace must contain DecisionTrace values")
        object.__setattr__(self, "trace", tuple(self.trace))
        object.__setattr__(
            self,
            "combinations_checked",
            _integer(self.combinations_checked, "combinations_checked"),
        )
        if not isinstance(self.record, ToyRunRecord):
            raise TypeError("record must be a ToyRunRecord")


def _validated_binary_vector(
    raw: object,
    *,
    count: int,
    lower: np.ndarray,
    upper: np.ndarray,
    matrix: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    try:
        vector = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None, "not numeric"
    if vector.ndim != 1 or len(vector) != count:
        return None, "wrong vector length"
    if not np.all(np.isfinite(vector)):
        return None, "non-finite variable"
    tolerance = 1e-7
    if np.any(vector < lower - tolerance) or np.any(vector > upper + tolerance):
        return None, "variable outside bounds"
    rounded = np.rint(vector)
    if np.any(np.abs(vector - rounded) > tolerance):
        return None, "non-binary variable"
    residual = matrix @ rounded
    if np.any(residual < constraint_lower - tolerance) or np.any(
        residual > constraint_upper + tolerance
    ):
        return None, "linear constraint violated"
    return rounded.astype(int), None


def default_batches() -> tuple[ThreatBatch, ...]:
    """Return the hand-audited three-release, one-resource instance."""

    return (
        ThreatBatch(
            "T1",
            0,
            (
                TaskPackage("T1-long", "T1", (0, 1), 8),
                TaskPackage("T1-short", "T1", (0,), 5),
            ),
        ),
        ThreatBatch(
            "T2",
            1,
            (
                TaskPackage("T2-long", "T2", (1, 2), 14),
                TaskPackage("T2-short", "T2", (1,), 8),
            ),
        ),
        ThreatBatch("T3", 2, (TaskPackage("T3-only", "T3", (2,), 5),)),
    )


def _validate_batches(batches: object) -> tuple[ThreatBatch, ...]:
    if not isinstance(batches, tuple):
        raise TypeError("batches must be a tuple")
    if not batches:
        raise ValueError("batches must not be empty")
    if any(not isinstance(batch, ThreatBatch) for batch in batches):
        raise TypeError("batches must contain ThreatBatch values")
    normalized = tuple(batches)
    ordered = tuple(
        sorted(normalized, key=lambda item: (item.release_time, item.threat_id))
    )
    if ordered != normalized:
        raise ValueError("batches must be sorted by release time and threat id")
    if len({batch.threat_id for batch in normalized}) != len(normalized):
        raise ValueError("threat ids must be unique")
    packages = tuple(package for batch in normalized for package in batch.packages)
    if len({package.package_id for package in packages}) != len(packages):
        raise ValueError("package ids must be globally unique")
    return normalized


def _packages(batches: tuple[ThreatBatch, ...]) -> tuple[TaskPackage, ...]:
    return tuple(package for batch in batches for package in batch.packages)


def verify_selection(
    batches: tuple[ThreatBatch, ...],
    selected_ids: tuple[str, ...],
) -> VerificationResult:
    """Independently verify package identity, threat, and slot constraints."""

    normalized = _validate_batches(batches)
    if not isinstance(selected_ids, tuple) or any(
        not isinstance(package_id, str) for package_id in selected_ids
    ):
        raise TypeError("selected_ids must be a tuple of strings")
    package_by_id = {package.package_id: package for package in _packages(normalized)}
    if len(set(selected_ids)) != len(selected_ids):
        return VerificationResult(False, 0.0, "selected package ids must be unique")
    unknown = next((item for item in selected_ids if item not in package_by_id), None)
    if unknown is not None:
        return VerificationResult(False, 0.0, f"unknown package id {unknown}")
    selected = tuple(package_by_id[item] for item in selected_ids)
    threat_counts: dict[str, int] = {}
    for package in selected:
        threat_counts[package.threat_id] = threat_counts.get(package.threat_id, 0) + 1
        if threat_counts[package.threat_id] > 1:
            return VerificationResult(
                False,
                0.0,
                f"more than one package selected for threat {package.threat_id}",
            )
    slot_counts: dict[int, int] = {}
    for package in selected:
        for slot in package.slots:
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
            if slot_counts[slot] > 1:
                return VerificationResult(False, 0.0, f"slot capacity exceeded at slot {slot}")
    return VerificationResult(True, sum(package.value for package in selected), None)


def _record(
    *,
    solver: str,
    seed: int,
    objective: float,
    runtime_s: float,
    converged: bool,
    passed: bool,
    failure: str | None,
    metadata: dict[str, Any],
) -> ToyRunRecord:
    return ToyRunRecord(
        demo_name="q4_synthetic_scheduling",
        solver=solver,
        seed=seed,
        objective=objective,
        runtime_s=runtime_s,
        converged=converged,
        passed_manual_case=passed,
        failure_reason=failure,
        metadata=metadata,
    )


def enumerate_offline(
    batches: tuple[ThreatBatch, ...],
    *,
    seed: int = 0,
) -> ScheduleResult:
    """Enumerate one choice (including none) per threat as an audit oracle."""

    normalized = _validate_batches(batches)
    normalized_seed = _seed(seed)
    started_at = perf_counter()
    best_ids: tuple[str, ...] = ()
    best_value = -math.inf
    checked = 0
    for choices in product(*(batch.packages + (None,) for batch in normalized)):
        checked += 1
        selected_ids = tuple(
            package.package_id for package in choices if package is not None
        )
        verification = verify_selection(normalized, selected_ids)
        if verification.valid and (
            verification.objective > best_value
            or (
                verification.objective == best_value
                and selected_ids < best_ids
            )
        ):
            best_ids = selected_ids
            best_value = verification.objective
    runtime_s = perf_counter() - started_at
    verification = verify_selection(normalized, best_ids)
    record = _record(
        solver="independent exhaustive enumeration",
        seed=normalized_seed,
        objective=verification.objective,
        runtime_s=runtime_s,
        converged=True,
        passed=verification.valid,
        failure=None,
        metadata={"combinations_checked": checked, "interpretation": "audit_oracle"},
    )
    return ScheduleResult(
        best_ids,
        verification.objective,
        True,
        False,
        verification.valid,
        None,
        record,
        combinations_checked=checked,
    )


def solve_offline_milp(
    batches: tuple[ThreatBatch, ...],
    *,
    seed: int = 0,
) -> ScheduleResult:
    """Solve the full-information benchmark; it is only a hindsight upper bound."""

    normalized = _validate_batches(batches)
    normalized_seed = _seed(seed)
    packages = _packages(normalized)
    threats = tuple(batch.threat_id for batch in normalized)
    slots = tuple(sorted({slot for package in packages for slot in package.slots}))
    rows = [
        [float(package.threat_id == threat) for package in packages] for threat in threats
    ]
    rows.extend([float(slot in package.slots) for package in packages] for slot in slots)
    matrix = np.asarray(rows)
    constraint_lower = np.zeros(len(rows))
    constraint_upper = np.ones(len(rows))
    lower_bounds = np.zeros(len(packages))
    upper_bounds = np.ones(len(packages))
    constraint = optimize.LinearConstraint(matrix, lb=constraint_lower, ub=constraint_upper)
    started_at = perf_counter()
    try:
        result = optimize.milp(
            c=-np.asarray([package.value for package in packages]),
            integrality=np.ones(len(packages)),
            bounds=optimize.Bounds(lower_bounds, upper_bounds),
            constraints=constraint,
            options={"time_limit": 10.0},
        )
    except Exception as error:  # scipy backends may raise before returning a status
        runtime_s = perf_counter() - started_at
        failure = f"milp_exception:{type(error).__name__}"
        return ScheduleResult(
            (),
            0.0,
            False,
            True,
            False,
            failure,
            _record(
                solver="scipy.optimize.milp",
                seed=normalized_seed,
                objective=0.0,
                runtime_s=runtime_s,
                converged=False,
                passed=False,
                failure=failure,
                metadata={"interpretation": "hindsight_upper_bound"},
            ),
        )
    runtime_s = perf_counter() - started_at
    if not result.success or result.x is None:
        failure = f"MILP failed: {result.message}"
        return ScheduleResult(
            (),
            0.0,
            False,
            True,
            False,
            failure,
            _record(
                solver="scipy.optimize.milp",
                seed=normalized_seed,
                objective=0.0,
                runtime_s=runtime_s,
                converged=False,
                passed=False,
                failure=failure,
                metadata={"interpretation": "hindsight_upper_bound"},
            ),
        )
    vector, vector_failure = _validated_binary_vector(
        result.x,
        count=len(packages),
        lower=lower_bounds,
        upper=upper_bounds,
        matrix=matrix,
        constraint_lower=constraint_lower,
        constraint_upper=constraint_upper,
    )
    if vector_failure is not None or vector is None:
        raise RuntimeError(f"invalid MILP solution: {vector_failure}")
    selected_ids = tuple(
        package.package_id
        for package, decision in zip(packages, vector, strict=True)
        if decision == 1
    )
    verification = verify_selection(normalized, selected_ids)
    oracle = enumerate_offline(normalized, seed=normalized_seed)
    if (
        not verification.valid
        or not math.isclose(verification.objective, oracle.objective, abs_tol=1e-8)
    ):
        raise RuntimeError("MILP/enumeration mismatch: hard verification failure")
    return ScheduleResult(
        selected_ids,
        verification.objective,
        True,
        False,
        True,
        None,
        _record(
            solver="scipy.optimize.milp",
            seed=normalized_seed,
            objective=verification.objective,
            runtime_s=runtime_s,
            converged=True,
            passed=True,
            failure=None,
            metadata={"interpretation": "hindsight_upper_bound", "oracle_match": True},
        ),
    )


def _causal_result(
    batches: tuple[ThreatBatch, ...],
    *,
    seed: int,
) -> ScheduleResult:
    normalized = _validate_batches(batches)
    normalized_seed = _seed(seed)
    started_at = perf_counter()
    selected: list[str] = []
    assigned_threats: set[str] = set()
    committed_slots: set[int] = set()
    trace: list[DecisionTrace] = []
    for now in range(max(batch.release_time for batch in normalized) + 1):
        visible = tuple(batch for batch in normalized if batch.release_time <= now)
        new_batches = tuple(
            batch
            for batch in visible
            if batch.release_time == now and batch.threat_id not in assigned_threats
        )
        feasible = tuple(
            package
            for batch in new_batches
            for package in batch.packages
            if not committed_slots.intersection(package.slots)
        )
        ordered_candidates = sorted(
            feasible,
            key=lambda item: (
                -(item.value / len(item.slots)),
                -item.value,
                item.package_id,
            ),
        )
        epoch_selected: list[str] = []
        for candidate in ordered_candidates:
            if (
                candidate.threat_id in assigned_threats
                or committed_slots.intersection(candidate.slots)
            ):
                continue
            selected.append(candidate.package_id)
            epoch_selected.append(candidate.package_id)
            assigned_threats.add(candidate.threat_id)
            committed_slots.update(candidate.slots)
        trace.append(
            DecisionTrace(
                now,
                tuple(batch.threat_id for batch in visible),
                tuple(epoch_selected),
                tuple(sorted(committed_slots)),
            )
        )
    verification = verify_selection(normalized, tuple(selected))
    runtime_s = perf_counter() - started_at
    solver = "causal value-density greedy"
    metadata = {
        "information": "released_threats_only",
        "whole_package_commitment": True,
        "rule": "value_density_then_value_then_id",
    }
    return ScheduleResult(
        tuple(selected),
        verification.objective,
        verification.valid,
        not verification.valid,
        verification.valid,
        verification.failure,
        _record(
            solver=solver,
            seed=normalized_seed,
            objective=verification.objective,
            runtime_s=runtime_s,
            converged=verification.valid,
            passed=verification.valid,
            failure=verification.failure,
            metadata=metadata,
        ),
        trace=tuple(trace),
    )


def solve_rolling_zero_forecast(
    batches: tuple[ThreatBatch, ...],
    *,
    seed: int = 0,
) -> ScheduleResult:
    """Re-solve a released-information MILP and commit whole packages each epoch."""

    normalized = _validate_batches(batches)
    normalized_seed = _seed(seed)
    started_at = perf_counter()
    selected: list[str] = []
    assigned_threats: set[str] = set()
    committed_slots: set[int] = set()
    trace: list[DecisionTrace] = []
    for now in range(max(batch.release_time for batch in normalized) + 1):
        visible = tuple(batch for batch in normalized if batch.release_time <= now)
        unresolved = tuple(
            batch for batch in visible if batch.threat_id not in assigned_threats
        )
        packages = tuple(package for batch in unresolved for package in batch.packages)
        if not packages:
            trace.append(
                DecisionTrace(
                    now,
                    tuple(batch.threat_id for batch in visible),
                    (),
                    tuple(sorted(committed_slots)),
                )
            )
            continue
        threats = tuple(batch.threat_id for batch in unresolved)
        slots = tuple(sorted({slot for package in packages for slot in package.slots}))
        rows = [
            [float(package.threat_id == threat) for package in packages]
            for threat in threats
        ]
        rows.extend(
            [float(slot in package.slots) for package in packages] for slot in slots
        )
        matrix = np.asarray(rows)
        constraint_lower = np.zeros(len(rows))
        constraint_upper = np.ones(len(rows))
        constraint = optimize.LinearConstraint(
            matrix,
            lb=constraint_lower,
            ub=constraint_upper,
        )
        lower_bounds = np.zeros(len(packages))
        upper_bounds = np.asarray(
            [
                float(
                    not committed_slots.intersection(package.slots)
                    and all(slot >= now for slot in package.slots)
                )
                for package in packages
            ]
        )
        try:
            result = optimize.milp(
                c=-np.asarray([package.value for package in packages]),
                integrality=np.ones(len(packages)),
                bounds=optimize.Bounds(lower_bounds, upper_bounds),
                constraints=constraint,
                options={"time_limit": 10.0},
            )
        except Exception as error:  # scipy backends may raise before returning a status
            runtime_s = perf_counter() - started_at
            failure = f"milp_exception:{type(error).__name__}"
            verification = verify_selection(normalized, tuple(selected))
            return ScheduleResult(
                tuple(selected),
                verification.objective,
                False,
                True,
                False,
                failure,
                _record(
                    solver="rolling zero-forecast MILP",
                    seed=normalized_seed,
                    objective=verification.objective,
                    runtime_s=runtime_s,
                    converged=False,
                    passed=False,
                    failure=failure,
                    metadata={
                        "information": "released_threats_only",
                        "failed_epoch": now,
                    },
                ),
                trace=tuple(trace),
            )
        if not result.success or result.x is None:
            runtime_s = perf_counter() - started_at
            failure = f"MILP failed at epoch {now}: {result.message}"
            verification = verify_selection(normalized, tuple(selected))
            return ScheduleResult(
                tuple(selected),
                verification.objective,
                False,
                True,
                False,
                failure,
                _record(
                    solver="rolling zero-forecast MILP",
                    seed=normalized_seed,
                    objective=verification.objective,
                    runtime_s=runtime_s,
                    converged=False,
                    passed=False,
                    failure=failure,
                    metadata={
                        "information": "released_threats_only",
                        "failed_epoch": now,
                    },
                ),
                trace=tuple(trace),
            )
        vector, vector_failure = _validated_binary_vector(
            result.x,
            count=len(packages),
            lower=lower_bounds,
            upper=upper_bounds,
            matrix=matrix,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
        )
        if vector_failure is not None or vector is None:
            runtime_s = perf_counter() - started_at
            failure = f"invalid_milp_solution:{vector_failure}"
            verification = verify_selection(normalized, tuple(selected))
            return ScheduleResult(
                tuple(selected),
                verification.objective,
                False,
                True,
                False,
                failure,
                _record(
                    solver="rolling zero-forecast MILP",
                    seed=normalized_seed,
                    objective=verification.objective,
                    runtime_s=runtime_s,
                    converged=False,
                    passed=False,
                    failure=failure,
                    metadata={
                        "information": "released_threats_only",
                        "failed_epoch": now,
                    },
                ),
                trace=tuple(trace),
            )
        epoch_selected = tuple(
            package
            for package, decision in zip(packages, vector, strict=True)
            if decision == 1
        )
        if any(
            package.threat_id not in {batch.threat_id for batch in visible}
            or any(slot < now for slot in package.slots)
            or committed_slots.intersection(package.slots)
            for package in epoch_selected
        ):
            raise RuntimeError("invalid MILP solution: causal information violation")
        for package in epoch_selected:
            selected.append(package.package_id)
            assigned_threats.add(package.threat_id)
            committed_slots.update(package.slots)
        trace.append(
            DecisionTrace(
                now,
                tuple(batch.threat_id for batch in visible),
                tuple(package.package_id for package in epoch_selected),
                tuple(sorted(committed_slots)),
            )
        )
    verification = verify_selection(normalized, tuple(selected))
    runtime_s = perf_counter() - started_at
    return ScheduleResult(
        tuple(selected),
        verification.objective,
        verification.valid,
        not verification.valid,
        verification.valid,
        verification.failure,
        _record(
            solver="rolling zero-forecast MILP",
            seed=normalized_seed,
            objective=verification.objective,
            runtime_s=runtime_s,
            converged=verification.valid,
            passed=verification.valid,
            failure=verification.failure,
            metadata={
                "information": "released_threats_only",
                "whole_package_commitment": True,
                "epochs_solved": len(trace),
            },
        ),
        trace=tuple(trace),
    )


def solve_causal_greedy(
    batches: tuple[ThreatBatch, ...],
    *,
    seed: int = 0,
) -> ScheduleResult:
    """Use a deterministic density/value/id rule on newly released packages."""

    return _causal_result(batches, seed=seed)


def run_demo(*, seed: int = 2026) -> dict[str, ScheduleResult]:
    """Run all four isolated comparisons without touching formal result paths."""

    normalized_seed = _seed(seed)
    batches = default_batches()
    return {
        "offline_enumeration": enumerate_offline(batches, seed=normalized_seed),
        "offline_milp": solve_offline_milp(batches, seed=normalized_seed),
        "rolling": solve_rolling_zero_forecast(batches, seed=normalized_seed),
        "greedy": solve_causal_greedy(batches, seed=normalized_seed),
    }
