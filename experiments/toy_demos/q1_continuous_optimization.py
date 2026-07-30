"""Budget-matched continuous-optimization toy benchmark for Q1 model selection.

This module deliberately uses a dimensionless artificial objective.  It does
not import any contest scenario, parameter, solver, or result.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import (
    Bounds,
    NonlinearConstraint,
    differential_evolution,
    minimize,
    shgo,
)
from scipy.stats import qmc

METHODS = (
    "multistart_slsqp",
    "de_slsqp",
    "pso_slsqp",
    "sobol_trust_constr",
    "shgo",
)
BOUNDS = ((-1.4, 1.4), (-1.0, 1.0))
KNOWN_OPTIMUM = (0.0, 0.0)
KNOWN_OBJECTIVE = 10.0
_VERIFICATION_TOLERANCE = 2e-5


def coverage(point: tuple[float, float] | np.ndarray) -> float:
    """Return the artificial coverage score maximized by the solvers."""

    x, y = (float(value) for value in point)
    loss = x * x + 1.4 * y * y
    loss += 0.18 * math.sin(5.0 * x) ** 2
    loss += 0.12 * math.sin(6.0 * y) ** 2
    return 10.0 - loss


def _ellipse_margin(point: tuple[float, float] | np.ndarray) -> float:
    x, y = (float(value) for value in point)
    return 1.0 - x * x / 1.4**2 - y * y


def _loss_gradient(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.asarray([2.0 * x + 0.9 * np.sin(10.0 * x), 2.8 * y + 0.72 * np.sin(12.0 * y)])


def _loss_hessian(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.diag([2.0 + 9.0 * np.cos(10.0 * x), 2.8 + 8.64 * np.cos(12.0 * y)])


def _margin_gradient(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.asarray([-2.0 * x / 1.4**2, -2.0 * y])


def _margin_hessian(point: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    del point
    return float(multiplier[0]) * np.diag([-2.0 / 1.4**2, -2.0])


@dataclass(frozen=True, slots=True)
class Verification:
    feasible: bool
    objective: float
    gap: float
    reported_error: float
    verified: bool


def verify_solution(
    point: tuple[float, float] | np.ndarray,
    *,
    reported_objective: float,
) -> Verification:
    """Independently recompute feasibility and value without the solver oracle."""

    x, y = (float(value) for value in point)
    feasible = (
        BOUNDS[0][0] - 1e-10 <= x <= BOUNDS[0][1] + 1e-10
        and BOUNDS[1][0] - 1e-10 <= y <= BOUNDS[1][1] + 1e-10
        and 1.0 - x * x / 1.4**2 - y * y >= -1e-9
    )
    independent_loss = (
        x**2
        + 1.4 * y**2
        + 0.18 * (1.0 - math.cos(10.0 * x)) / 2.0
        + 0.12 * (1.0 - math.cos(12.0 * y)) / 2.0
    )
    independent_objective = 10.0 - independent_loss
    gap = max(0.0, 10.0 - independent_objective)
    reported_error = abs(float(reported_objective) - independent_objective)
    verified = feasible and gap <= _VERIFICATION_TOLERANCE and reported_error <= 1e-8
    return Verification(
        feasible=feasible,
        objective=independent_objective,
        gap=gap,
        reported_error=reported_error,
        verified=verified,
    )


class _BudgetExhausted(RuntimeError):
    pass


class _CachedOracle:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.cache: dict[tuple[float, float], float] = {}

    @property
    def evaluation_count(self) -> int:
        return len(self.cache)

    def loss(self, point: np.ndarray) -> float:
        key = tuple(float(value) for value in np.asarray(point, dtype=float))
        if key not in self.cache:
            if self.evaluation_count >= self.budget:
                raise _BudgetExhausted
            self.cache[key] = 10.0 - coverage(key)
        return self.cache[key]

    def best_point(self) -> np.ndarray:
        if not self.cache:
            raise RuntimeError("no evaluated incumbent")
        return np.asarray(min(self.cache, key=self.cache.__getitem__), dtype=float)


@dataclass(frozen=True, slots=True)
class MethodResult:
    method: str
    x: tuple[float, float]
    objective: float
    evaluation_count: int
    budget: int
    solver_success: bool
    budget_exhausted: bool
    verified: bool
    gap: float
    failure: str | None
    runtime_s: float

    def as_dict(self, *, exclude_runtime: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload["x"] = list(self.x)
        if exclude_runtime:
            payload.pop("runtime_s")
        return payload


def _constraints() -> tuple[dict[str, Callable[[np.ndarray], float]], ...]:
    return ({"type": "ineq", "fun": _ellipse_margin},)


def _polish_slsqp(oracle: _CachedOracle, start: np.ndarray) -> Any:
    return minimize(
        oracle.loss,
        start,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=_constraints(),
        options={"ftol": 1e-12, "maxiter": 80},
    )


def _multistart_slsqp(oracle: _CachedOracle, seed: int) -> bool:
    starts = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(m=3)
    starts = qmc.scale(starts, [item[0] for item in BOUNDS], [item[1] for item in BOUNDS])
    successes = []
    for start in starts:
        if _ellipse_margin(start) >= 0.0:
            successes.append(bool(_polish_slsqp(oracle, start).success))
    return any(successes)


def _de_slsqp(oracle: _CachedOracle, seed: int) -> bool:
    result = differential_evolution(
        oracle.loss,
        BOUNDS,
        seed=seed,
        popsize=8,
        maxiter=14,
        polish=False,
        tol=1e-8,
        constraints=NonlinearConstraint(_ellipse_margin, 0.0, np.inf),
        updating="immediate",
        workers=1,
    )
    polished = _polish_slsqp(oracle, np.asarray(result.x))
    return bool(polished.success)


def _pso_slsqp(oracle: _CachedOracle, seed: int) -> bool:
    rng = np.random.default_rng(seed)
    particle_count = 18
    positions = rng.uniform(
        [item[0] for item in BOUNDS],
        [item[1] for item in BOUNDS],
        size=(particle_count, 2),
    )
    velocities = np.zeros_like(positions)
    personal = positions.copy()
    personal_values = np.full(particle_count, np.inf)
    global_best = positions[0].copy()
    global_value = np.inf
    for _ in range(18):
        for index, position in enumerate(positions):
            value = oracle.loss(position)
            if _ellipse_margin(position) < 0.0:
                value += 1_000.0 * (-_ellipse_margin(position))
            if value < personal_values[index]:
                personal[index] = position
                personal_values[index] = value
            if value < global_value:
                global_best = position.copy()
                global_value = value
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = (
            0.65 * velocities
            + 1.5 * r1 * (personal - positions)
            + 1.5 * r2 * (global_best - positions)
        )
        positions = np.clip(
            positions + velocities,
            [item[0] for item in BOUNDS],
            [item[1] for item in BOUNDS],
        )
    polished = _polish_slsqp(oracle, global_best)
    return bool(polished.success)


def _sobol_trust_constr(oracle: _CachedOracle, seed: int) -> bool:
    samples = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(m=7)
    samples = qmc.scale(samples, [item[0] for item in BOUNDS], [item[1] for item in BOUNDS])
    feasible_samples = [sample for sample in samples if _ellipse_margin(sample) >= 0.0]
    start = min(feasible_samples, key=oracle.loss)
    result = minimize(
        oracle.loss,
        start,
        method="trust-constr",
        jac=_loss_gradient,
        hess=_loss_hessian,
        bounds=Bounds(*zip(*BOUNDS, strict=True)),
        constraints=NonlinearConstraint(
            _ellipse_margin,
            0.0,
            np.inf,
            jac=_margin_gradient,
            hess=_margin_hessian,
        ),
        options={"gtol": 1e-10, "maxiter": 100},
    )
    polished = _polish_slsqp(oracle, np.asarray(result.x))
    return bool(result.success or polished.success)


def _shgo(oracle: _CachedOracle, seed: int) -> bool:
    del seed
    result = shgo(
        oracle.loss,
        BOUNDS,
        constraints=_constraints(),
        n=128,
        iters=1,
        sampling_method="simplicial",
        options={"minimize_every_iter": True},
    )
    polished = _polish_slsqp(oracle, np.asarray(result.x))
    return bool(result.success or polished.success)


_RUNNERS = {
    "multistart_slsqp": _multistart_slsqp,
    "de_slsqp": _de_slsqp,
    "pso_slsqp": _pso_slsqp,
    "sobol_trust_constr": _sobol_trust_constr,
    "shgo": _shgo,
}


def run_method(method: str, *, seed: int, budget: int = 768) -> MethodResult:
    """Run one solver under the shared unique-point evaluation budget."""

    if method not in _RUNNERS:
        raise ValueError(f"unknown method: {method}")
    if budget < 1:
        raise ValueError("budget must be positive")

    oracle = _CachedOracle(budget)
    started_at = perf_counter()
    solver_success = False
    exhausted = False
    try:
        solver_success = _RUNNERS[method](oracle, seed)
    except _BudgetExhausted:
        exhausted = True

    point = oracle.best_point()
    objective = coverage(point)
    verification = verify_solution(point, reported_objective=objective)
    failure = None
    if exhausted:
        solver_success = False
        failure = "evaluation_budget_exhausted"
    elif not solver_success:
        failure = "solver_did_not_converge"
    elif not verification.verified:
        failure = "independent_verification_failed"

    return MethodResult(
        method=method,
        x=(float(point[0]), float(point[1])),
        objective=objective,
        evaluation_count=oracle.evaluation_count,
        budget=budget,
        solver_success=solver_success and verification.verified,
        budget_exhausted=exhausted,
        verified=verification.verified and not exhausted,
        gap=verification.gap,
        failure=failure,
        runtime_s=perf_counter() - started_at,
    )


def run_demo(*, seed: int = 20260731, budget: int = 768) -> dict[str, MethodResult]:
    """Run all five methods with identical seed and evaluation budget."""

    return {method: run_method(method, seed=seed, budget=budget) for method in METHODS}
