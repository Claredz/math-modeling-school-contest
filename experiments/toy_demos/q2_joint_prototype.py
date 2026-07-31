"""Synthetic joint discrete--continuous Q2 scheduling demonstration.

This module deliberately lives outside the formal model.  Three distinct toy
bomb types may each be used at most once on ``t in [0, 4]``.  Selecting and
ordering bomb types is discrete; their burst times are continuous and must be
at least 0.5 time units apart.  Coverage is a sum of triangular kernels and
the objective is the worst continuous-time coverage-to-demand ratio.

Two heuristic routes are compared:

* candidate combinations followed by continuous SLSQP polishing;
* local finite masters followed by an exact separation oracle.

Neither route's local-master value is presented as a global upper bound.  The
only global upper bound returned here follows from a quarter-grid incumbent
and a proved Lipschitz covering argument.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from time import perf_counter
from types import MappingProxyType

import numpy as np
from scipy.optimize import minimize

from experiments.toy_demos.common import ToyRunRecord, seeded_rng

HORIZON = 4.0
MIN_SEPARATION = 0.5
GRID_BOUND_SOURCE = "Lipschitz grid covering bound"
DEMAND_NODES: tuple[tuple[float, float], ...] = (
    (0.0, 0.7),
    (1.0, 1.1),
    (2.0, 0.85),
    (3.0, 1.2),
    (4.0, 0.75),
)
BOMB_PARAMETERS = MappingProxyType(
    {
        "A": (1.0, 1.2),
        "B": (0.9, 1.5),
        "C": (1.1, 1.0),
    }
)


@dataclass(frozen=True, slots=True)
class BombSchedule:
    """A discrete bomb order and its continuous burst-time vector."""

    bomb_types: tuple[str, ...]
    burst_times: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            bomb_types = tuple(self.bomb_types)
            raw_times = tuple(self.burst_times)
        except TypeError as error:
            raise TypeError("bomb types and burst times must be sequences") from error
        if not 1 <= len(bomb_types) <= 3:
            raise ValueError("a schedule must use between one and three bombs")
        if len(bomb_types) != len(raw_times):
            raise ValueError("bomb type and burst-time counts must match")
        if any(not isinstance(name, str) for name in bomb_types):
            raise TypeError("bomb type names must be strings")
        if len(set(bomb_types)) != len(bomb_types):
            raise ValueError("each toy bomb type may be used at most once")
        if any(name not in BOMB_PARAMETERS for name in bomb_types):
            raise ValueError("unknown toy bomb type")

        times: list[float] = []
        for value in raw_times:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("burst times must be real numbers")
            time = float(value)
            if not math.isfinite(time) or not 0.0 <= time <= HORIZON:
                raise ValueError("burst times must be finite and inside the horizon")
            times.append(time)
        if any(
            later - earlier < MIN_SEPARATION - 1e-12
            for earlier, later in zip(times, times[1:], strict=False)
        ):
            raise ValueError("successive bursts must be separated by at least 0.5")
        object.__setattr__(self, "bomb_types", bomb_types)
        object.__setattr__(self, "burst_times", tuple(times))


@dataclass(frozen=True, slots=True)
class ExactVerification:
    """Exact continuous-time objective obtained from all linear breakpoints."""

    objective: float
    worst_time: float
    breakpoints: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GridBounds:
    """Feasible grid incumbent and a Lipschitz-certified global coarse bound."""

    schedule: BombSchedule
    lower_bound: float
    global_upper_bound: float
    grid_step: float
    lipschitz_constant: float
    evaluated_schedules: int
    bound_source: str = GRID_BOUND_SOURCE

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, BombSchedule):
            raise TypeError("schedule must be a BombSchedule")
        numeric_fields = (
            "lower_bound",
            "global_upper_bound",
            "grid_step",
            "lipschitz_constant",
        )
        normalized: dict[str, float] = {}
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
            normalized[field_name] = number
            object.__setattr__(self, field_name, number)
        if normalized["grid_step"] <= 0.0:
            raise ValueError("grid_step must be positive")
        _validate_grid_step(normalized["grid_step"])
        if isinstance(self.evaluated_schedules, bool) or not isinstance(
            self.evaluated_schedules, Integral
        ):
            raise TypeError("evaluated_schedules must be an integer")
        if int(self.evaluated_schedules) <= 0:
            raise ValueError("evaluated_schedules must be positive")
        object.__setattr__(self, "evaluated_schedules", int(self.evaluated_schedules))
        if self.bound_source != GRID_BOUND_SOURCE:
            raise ValueError("unrecognised global-bound provenance")
        exact_objective = verify_schedule_exactly(self.schedule).objective
        if not math.isclose(
            normalized["lower_bound"], exact_objective, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError("lower_bound must equal the exact objective of schedule")
        expected_lipschitz = _objective_lipschitz_constant()
        if not math.isclose(
            normalized["lipschitz_constant"],
            expected_lipschitz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("lipschitz_constant does not match the proved constant")
        expected_upper = normalized["lower_bound"] + expected_lipschitz * normalized["grid_step"]
        if not math.isclose(
            normalized["global_upper_bound"],
            expected_upper,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("global_upper_bound does not match its provenance formula")
        if normalized["lower_bound"] > normalized["global_upper_bound"]:
            raise ValueError("lower_bound must not exceed global_upper_bound")


@dataclass(frozen=True, slots=True)
class RouteResult:
    """Auditable output of one joint discrete--continuous heuristic."""

    route_name: str
    schedule: BombSchedule
    verified_objective: float
    converged: bool
    local_solver_converged: bool
    globally_resolved: bool
    unresolved: bool
    iterations: int
    failure_reason: str | None
    global_lower_bound: float
    global_upper_bound: float
    global_gap: float
    master_values: tuple[float, ...]
    record: ToyRunRecord


@dataclass(frozen=True, slots=True)
class _LocalSolveOutcome:
    schedule: BombSchedule
    value: float
    success: bool
    status: int
    message: str


@dataclass(frozen=True, slots=True)
class JointDemoResult:
    """Results of the shared baseline and both independently verified routes."""

    seed: int
    grid_bounds: GridBounds
    candidate_route: RouteResult
    oracle_route: RouteResult


def _validated_schedule(schedule: BombSchedule) -> BombSchedule:
    if not isinstance(schedule, BombSchedule):
        raise TypeError("schedule must be a BombSchedule")
    if not 1 <= len(schedule.bomb_types) <= 3:
        raise ValueError("a schedule must use between one and three bombs")
    if len(schedule.bomb_types) != len(schedule.burst_times):
        raise ValueError("bomb type and burst-time counts must match")
    if len(set(schedule.bomb_types)) != len(schedule.bomb_types):
        raise ValueError("each toy bomb type may be used at most once")
    if any(name not in BOMB_PARAMETERS for name in schedule.bomb_types):
        raise ValueError("unknown toy bomb type")

    times: list[float] = []
    for value in schedule.burst_times:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("burst times must be real numbers")
        time = float(value)
        if not math.isfinite(time) or not 0.0 <= time <= HORIZON:
            raise ValueError("burst times must be finite and inside the horizon")
        times.append(time)
    if any(
        later - earlier < MIN_SEPARATION - 1e-12
        for earlier, later in zip(times, times[1:], strict=False)
    ):
        raise ValueError("successive bursts must be separated by at least 0.5")
    return BombSchedule(tuple(schedule.bomb_types), tuple(times))


def _demand(time: float) -> float:
    for (left_t, left_d), (right_t, right_d) in zip(
        DEMAND_NODES, DEMAND_NODES[1:], strict=False
    ):
        if time <= right_t + 1e-14:
            fraction = (time - left_t) / (right_t - left_t)
            return left_d + fraction * (right_d - left_d)
    return DEMAND_NODES[-1][1]


def evaluate_coverage(schedule: BombSchedule, time: float) -> float:
    """Evaluate the nonnegative sum of triangular kernels at one time."""

    checked = _validated_schedule(schedule)
    if isinstance(time, bool) or not isinstance(time, (int, float, np.integer, np.floating)):
        raise TypeError("time must be a real number")
    normalized_time = float(time)
    if not math.isfinite(normalized_time) or not 0.0 <= normalized_time <= HORIZON:
        raise ValueError("time must be finite and inside the horizon")

    return _coverage_unchecked(checked, normalized_time)


def _coverage_unchecked(schedule: BombSchedule, time: float) -> float:
    """Hot-path kernel evaluation for already validated/optimizer trial vectors."""

    return _coverage_from_vectors(schedule.bomb_types, schedule.burst_times, time)


def _coverage_from_vectors(
    bomb_types: Sequence[str],
    burst_times: Sequence[float],
    time: float,
) -> float:
    """Evaluate kernels for optimizer trial vectors without public validation."""

    coverage = 0.0
    for bomb_type, burst_time in zip(
        bomb_types, burst_times, strict=True
    ):
        half_width, peak = BOMB_PARAMETERS[bomb_type]
        coverage += peak * max(0.0, 1.0 - abs(time - burst_time) / half_width)
    return coverage


def _all_breakpoints(schedule: BombSchedule) -> tuple[float, ...]:
    points = {time for time, _ in DEMAND_NODES}
    for bomb_type, burst_time in zip(schedule.bomb_types, schedule.burst_times, strict=True):
        half_width, _ = BOMB_PARAMETERS[bomb_type]
        points.update((burst_time - half_width, burst_time, burst_time + half_width))
    clipped = sorted(min(HORIZON, max(0.0, point)) for point in points)
    unique: list[float] = []
    for point in clipped:
        if not unique or abs(point - unique[-1]) > 1e-12:
            unique.append(point)
    return tuple(unique)


def verify_schedule_exactly(schedule: BombSchedule) -> ExactVerification:
    """Compute the exact minimum ratio over the continuous horizon.

    Coverage and demand are affine between consecutive combined breakpoints.
    The ratio of two affine functions with positive denominator is monotone or
    constant, so its minimum lies at a breakpoint.
    """

    checked = _validated_schedule(schedule)
    breakpoints = _all_breakpoints(checked)
    ratios = tuple(evaluate_coverage(checked, time) / _demand(time) for time in breakpoints)
    worst_index = min(range(len(ratios)), key=lambda index: (ratios[index], breakpoints[index]))
    return ExactVerification(
        objective=float(ratios[worst_index]),
        worst_time=breakpoints[worst_index],
        breakpoints=breakpoints,
    )


def sample_objective(schedule: BombSchedule, *, sample_times: Iterable[float]) -> float:
    """Return a deliberately finite-sample objective for audit comparisons."""

    checked = _validated_schedule(schedule)
    raw_times = tuple(sample_times)
    if any(isinstance(time, bool) or not isinstance(time, Real) for time in raw_times):
        raise TypeError("sample times must be real numbers")
    times = tuple(float(time) for time in raw_times)
    if not times:
        raise ValueError("at least one sample time is required")
    if any(not math.isfinite(time) or not 0.0 <= time <= HORIZON for time in times):
        raise ValueError("sample times must be finite and inside the horizon")
    return min(evaluate_coverage(checked, time) / _demand(time) for time in times)


def _discrete_modes() -> tuple[tuple[str, ...], ...]:
    names = tuple(BOMB_PARAMETERS)
    return tuple(
        permutation
        for count in range(1, len(names) + 1)
        for subset in itertools.combinations(names, count)
        for permutation in itertools.permutations(subset)
    )


def _feasible_grid_times(count: int, grid_step: float) -> Iterable[tuple[float, ...]]:
    number_of_steps = round(HORIZON / grid_step)
    indices = range(number_of_steps + 1)
    minimum_index_gap = round(MIN_SEPARATION / grid_step)
    for combination in itertools.combinations(indices, count):
        if all(
            later - earlier >= minimum_index_gap
            for earlier, later in zip(combination, combination[1:], strict=False)
        ):
            yield tuple(index * grid_step for index in combination)


def _validate_grid_step(grid_step: float) -> float:
    if isinstance(grid_step, bool) or not isinstance(
        grid_step, (int, float, np.integer, np.floating)
    ):
        raise TypeError("grid_step must be a real number")
    step = float(grid_step)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("grid_step must be positive and finite")
    horizon_steps = HORIZON / step
    separation_steps = MIN_SEPARATION / step
    if abs(horizon_steps - round(horizon_steps)) > 1e-10:
        raise ValueError("grid_step must divide the horizon")
    if abs(separation_steps - round(separation_steps)) > 1e-10:
        raise ValueError("grid_step must divide the minimum separation")
    return step


def _objective_lipschitz_constant() -> float:
    minimum_demand = min(value for _, value in DEMAND_NODES)
    return sum(
        peak / half_width for half_width, peak in BOMB_PARAMETERS.values()
    ) / minimum_demand


@lru_cache(maxsize=16)
def enumerate_grid_bounds(*, grid_step: float = 0.25) -> GridBounds:
    """Exhaust a feasible time grid and certify a global coarse upper bound.

    Flooring any feasible continuous time to this grid preserves order and the
    0.5 separation because the grid divides 0.5.  Each kernel is Lipschitz in
    its burst time with constant ``peak / half_width``.  Dividing their sum by
    the minimum positive demand gives the objective Lipschitz constant.
    """

    step = _validate_grid_step(grid_step)
    best_schedule: BombSchedule | None = None
    best_objective = -math.inf
    evaluated = 0
    for mode in _discrete_modes():
        for times in _feasible_grid_times(len(mode), step):
            schedule = BombSchedule(mode, times)
            objective = verify_schedule_exactly(schedule).objective
            evaluated += 1
            key = (objective, tuple(-time for time in times), mode)
            incumbent_key = (
                best_objective,
                tuple(-time for time in best_schedule.burst_times)
                if best_schedule is not None
                else (),
                best_schedule.bomb_types if best_schedule is not None else (),
            )
            if best_schedule is None or key > incumbent_key:
                best_schedule = schedule
                best_objective = objective
    if best_schedule is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("grid enumeration produced no feasible schedule")

    lipschitz_constant = _objective_lipschitz_constant()
    return GridBounds(
        schedule=best_schedule,
        lower_bound=best_objective,
        global_upper_bound=best_objective + lipschitz_constant * step,
        grid_step=step,
        lipschitz_constant=lipschitz_constant,
        evaluated_schedules=evaluated,
        bound_source=GRID_BOUND_SOURCE,
    )


def _expected_grid_schedule_count(grid_step: float) -> int:
    """Count the exhaustive schedule grid without trusting a supplied certificate."""

    step = _validate_grid_step(grid_step)
    point_count = round(HORIZON / step) + 1
    minimum_index_gap = round(MIN_SEPARATION / step)
    return sum(
        math.perm(len(BOMB_PARAMETERS), bomb_count)
        * math.comb(
            point_count - (bomb_count - 1) * (minimum_index_gap - 1),
            bomb_count,
        )
        for bomb_count in range(1, len(BOMB_PARAMETERS) + 1)
    )


def _recompute_and_validate_global_bounds(
    global_bounds: GridBounds | None,
) -> GridBounds:
    """Accept only a certificate identical to a cached exhaustive recomputation."""

    if global_bounds is None:
        return enumerate_grid_bounds(grid_step=0.25)
    if not isinstance(global_bounds, GridBounds):
        raise TypeError("global_bounds must be a GridBounds")
    expected_count = _expected_grid_schedule_count(global_bounds.grid_step)
    if global_bounds.evaluated_schedules != expected_count:
        raise ValueError(
            "global_bounds differs from the recomputed exhaustive grid optimum"
        )
    recomputed = enumerate_grid_bounds(grid_step=global_bounds.grid_step)
    if global_bounds != recomputed:
        raise ValueError(
            "global_bounds differs from the recomputed exhaustive grid optimum"
        )
    return recomputed


def _candidate_times() -> tuple[float, ...]:
    nodes = tuple(time for time, _ in DEMAND_NODES)
    midpoints = tuple(
        (left + right) / 2.0 for left, right in zip(nodes, nodes[1:], strict=False)
    )
    return tuple(sorted((*nodes, *midpoints)))


def _feasible_candidate_times(count: int) -> Iterable[tuple[float, ...]]:
    for times in itertools.combinations(_candidate_times(), count):
        if all(
            later - earlier >= MIN_SEPARATION
            for earlier, later in zip(times, times[1:], strict=False)
        ):
            yield times


def _polish_exact_objective(
    mode: tuple[str, ...],
    initial_times: Sequence[float],
    *,
    maxiter: int = 120,
) -> _LocalSolveOutcome:
    count = len(mode)

    def loss(values: np.ndarray) -> float:
        try:
            schedule = BombSchedule(mode, tuple(float(value) for value in values))
            return -verify_schedule_exactly(schedule).objective
        except (TypeError, ValueError):
            return 1e6

    constraints = [
        {
            "type": "ineq",
            "fun": lambda values, index=index: values[index + 1]
            - values[index]
            - MIN_SEPARATION,
        }
        for index in range(count - 1)
    ]
    optimized = minimize(
        loss,
        np.asarray(initial_times, dtype=float),
        method="SLSQP",
        bounds=[(0.0, HORIZON)] * count,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": maxiter, "disp": False},
    )
    values = np.asarray(optimized.x, dtype=float)
    values = np.clip(values, 0.0, HORIZON)
    for index in range(1, count):
        values[index] = max(values[index], values[index - 1] + MIN_SEPARATION)
    if values[-1] > HORIZON:
        shift = values[-1] - HORIZON
        values -= shift
    try:
        schedule = BombSchedule(mode, tuple(float(value) for value in values))
    except (TypeError, ValueError):
        schedule = BombSchedule(mode, tuple(float(value) for value in initial_times))
        return _LocalSolveOutcome(
            schedule=schedule,
            value=verify_schedule_exactly(schedule).objective,
            success=False,
            status=int(optimized.status),
            message=f"{optimized.message}; optimizer returned an infeasible schedule",
        )
    return _LocalSolveOutcome(
        schedule=schedule,
        value=verify_schedule_exactly(schedule).objective,
        success=bool(optimized.success),
        status=int(optimized.status),
        message=str(optimized.message),
    )


def _make_route_result(
    *,
    route_name: str,
    schedule: BombSchedule,
    local_solver_converged: bool,
    algorithm_resolved: bool,
    iterations: int,
    failure_reason: str | None,
    grid_bounds: GridBounds,
    master_values: Sequence[float],
    seed: int,
    runtime_s: float,
    solver_statuses: Sequence[str],
    resolution_tolerance: float = 1e-7,
) -> RouteResult:
    verification = verify_schedule_exactly(schedule)
    lower_bound = max(grid_bounds.lower_bound, verification.objective)
    gap = grid_bounds.global_upper_bound - lower_bound
    if gap < -1e-10:
        raise RuntimeError("verified lower bound exceeds the certified global upper bound")
    if gap < 0.0:
        gap = 0.0
    globally_resolved = (
        local_solver_converged and algorithm_resolved and gap <= resolution_tolerance
    )
    unresolved = not globally_resolved
    if unresolved and failure_reason is None:
        failure_reason = "global Lipschitz bound gap remains unresolved"
    record = ToyRunRecord(
        demo_name="q2_joint_discrete_continuous",
        solver=route_name,
        seed=seed,
        objective=verification.objective,
        runtime_s=runtime_s,
        converged=globally_resolved,
        passed_manual_case=(
            grid_bounds.lower_bound <= verification.objective + 1e-8
            and verification.objective <= grid_bounds.global_upper_bound + 1e-8
        ),
        failure_reason=failure_reason,
        metadata={
            "bomb_types": list(schedule.bomb_types),
            "burst_times": list(schedule.burst_times),
            "grid_lower_bound": grid_bounds.lower_bound,
            "global_lower_bound": lower_bound,
            "global_upper_bound": grid_bounds.global_upper_bound,
            "global_gap": gap,
            "bound_source": GRID_BOUND_SOURCE,
            "master_is_global_upper_bound": False,
            "local_solver_converged": local_solver_converged,
            "globally_resolved": globally_resolved,
            "unresolved": unresolved,
            "solver_statuses": list(solver_statuses),
            "master_values": list(master_values),
        },
    )
    return RouteResult(
        route_name=route_name,
        schedule=schedule,
        verified_objective=verification.objective,
        converged=globally_resolved,
        local_solver_converged=local_solver_converged,
        globally_resolved=globally_resolved,
        unresolved=unresolved,
        iterations=iterations,
        failure_reason=failure_reason,
        global_lower_bound=lower_bound,
        global_upper_bound=grid_bounds.global_upper_bound,
        global_gap=gap,
        master_values=tuple(float(value) for value in master_values),
        record=record,
    )


def solve_candidate_polish(
    *,
    seed: int = 0,
    global_bounds: GridBounds | None = None,
) -> RouteResult:
    """Search candidate combinations, then polish burst times with SLSQP."""

    rng = seeded_rng(seed)
    bounds = _recompute_and_validate_global_bounds(global_bounds)
    started = perf_counter()

    candidates: list[tuple[float, BombSchedule]] = [
        (bounds.lower_bound, bounds.schedule)
    ]
    modes = list(_discrete_modes())
    rng.shuffle(modes)
    for mode in modes:
        best_for_mode: tuple[float, BombSchedule] | None = None
        for times in _feasible_candidate_times(len(mode)):
            schedule = BombSchedule(mode, times)
            objective = verify_schedule_exactly(schedule).objective
            if best_for_mode is None or objective > best_for_mode[0]:
                best_for_mode = (objective, schedule)
        if best_for_mode is not None:
            candidates.append(best_for_mode)

    ranked = sorted(
        candidates,
        key=lambda item: (item[0], item[1].bomb_types, item[1].burst_times),
        reverse=True,
    )
    master_values = [objective for objective, _ in ranked[:8]]
    best_objective = bounds.lower_bound
    best_schedule = bounds.schedule
    outcomes: list[_LocalSolveOutcome] = []
    for _, candidate in ranked[:8]:
        outcome = _polish_exact_objective(candidate.bomb_types, candidate.burst_times)
        outcomes.append(outcome)
        polished = outcome.schedule
        if (outcome.value, polished.bomb_types, polished.burst_times) > (
            best_objective,
            best_schedule.bomb_types,
            best_schedule.burst_times,
        ):
            best_objective = outcome.value
            best_schedule = polished
    failed_statuses = tuple(
        f"status={outcome.status}: {outcome.message}"
        for outcome in outcomes
        if not outcome.success
    )
    local_solver_converged = bool(outcomes) and not failed_statuses

    return _make_route_result(
        route_name="candidate-combinations + SLSQP polish",
        schedule=best_schedule,
        local_solver_converged=local_solver_converged,
        algorithm_resolved=True,
        iterations=len(ranked[:8]),
        failure_reason="; ".join(failed_statuses) if failed_statuses else None,
        grid_bounds=bounds,
        master_values=master_values,
        seed=seed,
        runtime_s=perf_counter() - started,
        solver_statuses=tuple(
            f"status={outcome.status}: {outcome.message}" for outcome in outcomes
        ),
    )


def _solve_finite_master_for_mode(
    mode: tuple[str, ...],
    witnesses: Sequence[float],
    initial_times: Sequence[float],
) -> _LocalSolveOutcome:
    count = len(mode)

    def loss(values: np.ndarray) -> float:
        return -float(values[-1])

    def witness_constraint(values: np.ndarray, witness: float) -> float:
        trial_times = tuple(float(value) for value in values[:-1])
        return (
            _coverage_from_vectors(mode, trial_times, witness) / _demand(witness)
            - float(values[-1])
        )

    constraints: list[dict[str, object]] = [
        {
            "type": "ineq",
            "fun": lambda values, index=index: values[index + 1]
            - values[index]
            - MIN_SEPARATION,
        }
        for index in range(count - 1)
    ]
    constraints.extend(
        {
            "type": "ineq",
            "fun": lambda values, witness=witness: witness_constraint(values, witness),
        }
        for witness in witnesses
    )
    initial_schedule = BombSchedule(mode, tuple(float(value) for value in initial_times))
    initial_z = sample_objective(initial_schedule, sample_times=witnesses)
    optimized = minimize(
        loss,
        np.asarray((*initial_times, max(0.0, initial_z)), dtype=float),
        method="SLSQP",
        bounds=[(0.0, HORIZON)] * count + [(0.0, 10.0)],
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 200, "disp": False},
    )
    raw_times = tuple(float(value) for value in optimized.x[:-1])
    success = bool(optimized.success)
    message = str(optimized.message)
    try:
        schedule = BombSchedule(mode, raw_times)
    except (TypeError, ValueError):
        schedule = initial_schedule
        success = False
        message = f"{message}; optimizer returned an infeasible schedule"
    master_value = sample_objective(schedule, sample_times=witnesses)
    return _LocalSolveOutcome(
        schedule=schedule,
        value=master_value,
        success=success,
        status=int(optimized.status),
        message=message,
    )


def _grid_start_for_mode(mode: tuple[str, ...], step: float) -> BombSchedule:
    best_schedule: BombSchedule | None = None
    best_objective = -math.inf
    for times in _feasible_grid_times(len(mode), step):
        schedule = BombSchedule(mode, times)
        objective = verify_schedule_exactly(schedule).objective
        if best_schedule is None or objective > best_objective:
            best_schedule = schedule
            best_objective = objective
    if best_schedule is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("no grid start for mode")
    return best_schedule


def solve_separation_oracle(
    *,
    seed: int = 0,
    global_bounds: GridBounds | None = None,
    max_iterations: int = 8,
    tolerance: float = 1e-7,
) -> RouteResult:
    """Alternate local finite masters and an exact breakpoint separator.

    ``converged`` means only that the selected local finite-master schedule has
    no missed breakpoint beyond ``tolerance``.  The separately reported global
    gap remains the Lipschitz grid gap.
    """

    rng = seeded_rng(seed)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a real number")
    normalized_tolerance = float(tolerance)
    if not math.isfinite(normalized_tolerance) or normalized_tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    bounds = _recompute_and_validate_global_bounds(global_bounds)
    started = perf_counter()

    witnesses: list[float] = [time for time, _ in DEMAND_NODES]
    modes = list(_discrete_modes())
    rng.shuffle(modes)
    starts = {mode: _grid_start_for_mode(mode, bounds.grid_step) for mode in modes}
    incumbent = bounds.schedule
    incumbent_objective = bounds.lower_bound
    master_values: list[float] = []
    separator_converged = False
    completed_iterations = 0
    outcomes: list[_LocalSolveOutcome] = []
    termination_reason: str | None = None

    for iteration in range(max_iterations + 1):
        solved: list[tuple[float, BombSchedule]] = []
        for mode in modes:
            outcome = _solve_finite_master_for_mode(
                mode, witnesses, starts[mode].burst_times
            )
            outcomes.append(outcome)
            schedule = outcome.schedule
            master_value = outcome.value
            starts[mode] = schedule
            solved.append((master_value, schedule))
            verified = verify_schedule_exactly(schedule)
            if verified.objective > incumbent_objective:
                incumbent = schedule
                incumbent_objective = verified.objective
        selected_master, selected_schedule = max(
            solved, key=lambda item: (item[0], item[1].bomb_types, item[1].burst_times)
        )
        master_values.append(selected_master)
        separated = verify_schedule_exactly(selected_schedule)
        completed_iterations = iteration
        if (
            iteration > 0
            and selected_master - separated.objective <= normalized_tolerance
        ):
            separator_converged = True
            if separated.objective > incumbent_objective:
                incumbent = selected_schedule
            break
        if iteration == max_iterations:
            termination_reason = "maximum separation-oracle iterations reached"
            break
        if all(abs(separated.worst_time - witness) > 1e-10 for witness in witnesses):
            witnesses.append(separated.worst_time)
            witnesses.sort()
        else:
            # A repeated witness indicates local-master stagnation, not a global proof.
            termination_reason = "separation oracle stagnated on a repeated witness"
            break

    failed_statuses = tuple(
        f"status={outcome.status}: {outcome.message}"
        for outcome in outcomes
        if not outcome.success
    )
    local_solver_converged = bool(outcomes) and not failed_statuses
    if failed_statuses:
        failure_reason = "; ".join(failed_statuses)
    elif not separator_converged:
        failure_reason = termination_reason
    else:
        failure_reason = None
    return _make_route_result(
        route_name="local finite-master + exact separation heuristic",
        schedule=incumbent,
        local_solver_converged=local_solver_converged,
        algorithm_resolved=separator_converged,
        iterations=completed_iterations,
        failure_reason=failure_reason,
        grid_bounds=bounds,
        master_values=master_values,
        seed=seed,
        runtime_s=perf_counter() - started,
        solver_statuses=tuple(
            f"status={outcome.status}: {outcome.message}" for outcome in outcomes
        ),
        resolution_tolerance=normalized_tolerance,
    )


def run_q2_joint_demo(*, seed: int = 0, max_iterations: int = 8) -> JointDemoResult:
    """Run the baseline and both routes without writing any formal artifact."""

    seeded_rng(seed)  # validate before doing the expensive baseline
    bounds = enumerate_grid_bounds(grid_step=0.25)
    return JointDemoResult(
        seed=seed,
        grid_bounds=bounds,
        candidate_route=solve_candidate_polish(seed=seed, global_bounds=bounds),
        oracle_route=solve_separation_oracle(
            seed=seed,
            global_bounds=bounds,
            max_iterations=max_iterations,
        ),
    )
