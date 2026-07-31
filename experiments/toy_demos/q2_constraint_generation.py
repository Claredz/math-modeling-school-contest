"""Synthetic Q2 semi-infinite covering problem solved by constraint generation.

The unit disk must be covered by two equal disks centred at ``(-a, 0)`` and
``(a, 0)``.  A finite master initially sees only three collinear witnesses,
so it incorrectly prefers ``a = r = 1/2``.  The exact separation oracle finds
the missing pole, after which the master converges to ``a = 0, r = 1``.

This module is an isolated algorithm demonstration.  It does not use the
competition instances or write formal result artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real

from experiments.toy_demos.common import ToyRunRecord

Point = tuple[float, float]
DEFAULT_WITNESSES: tuple[Point, ...] = ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))


@dataclass(frozen=True, slots=True)
class CoverSolution:
    """Finite-master solution; ``objective`` is the squared common radius."""

    a: float
    radius: float
    objective: float


@dataclass(frozen=True, slots=True)
class SeparationResult:
    """Exact maximum power-distance violation over the unit disk."""

    witness: Point
    violation: float
    is_violated: bool


@dataclass(frozen=True, slots=True)
class IndependentVerification:
    """Result from a verifier that does not call the separation oracle."""

    passed: bool
    maximum_power_violation: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class ConstraintGenerationResult:
    """Deterministic trace of the constraint-generation loop.

    ``current_violations`` preserves every iterate's exact residual.
    ``best_seen_positive_violations`` is its non-increasing running minimum;
    it describes the best evaluated iterate and is not the current violation.
    """

    solution: CoverSolution
    converged: bool
    augmentations: int
    master_solves: int
    oracle_calls: int
    initial_witnesses: tuple[Point, ...]
    added_witnesses: tuple[Point, ...]
    current_violations: tuple[float, ...]
    best_seen_positive_violations: tuple[float, ...]
    seed: int
    failure_reason: str | None

    @property
    def iterations(self) -> int:
        """Backward-compatible alias for the number of witness augmentations."""

        return self.augmentations

    @property
    def violation_upper_bounds(self) -> tuple[float, ...]:
        """Deprecated compatibility alias for best-seen positive violations."""

        return self.best_seen_positive_violations

    def to_toy_record(self, *, runtime_s: float) -> ToyRunRecord:
        """Adapt the domain trace to the shared, strict result contract."""

        manual_case_passed = (
            self.converged
            and abs(self.solution.a) <= 1e-9
            and abs(self.solution.radius - 1.0) <= 1e-9
        )
        return ToyRunRecord(
            demo_name="q2_constraint_generation",
            solver="exact_symmetric_master_plus_power_distance_oracle",
            seed=self.seed,
            objective=self.solution.objective,
            runtime_s=runtime_s,
            converged=self.converged,
            passed_manual_case=manual_case_passed,
            failure_reason=self.failure_reason,
            metadata={
                "seed_role": "audit_only_deterministic_solver",
                "augmentations": self.augmentations,
                "master_solves": self.master_solves,
                "oracle_calls": self.oracle_calls,
                "current_violations": self.current_violations,
                "best_seen_positive_violations": self.best_seen_positive_violations,
                "added_witnesses": self.added_witnesses,
            },
        )


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return normalized


def _nonnegative_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return normalized


def _validate_solution(solution: CoverSolution) -> None:
    a = _nonnegative_finite(solution.a, "solution.a")
    radius = _nonnegative_finite(solution.radius, "solution.radius")
    objective = _nonnegative_finite(solution.objective, "solution.objective")
    if a > 1.0 + 1e-12:
        raise ValueError("solution.a must not exceed the unit-disk radius")
    if not math.isclose(objective, radius * radius, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("solution.objective must equal radius squared")


def _normalise_witnesses(witnesses: Iterable[Point]) -> tuple[Point, ...]:
    normalised = tuple((float(x), float(y)) for x, y in witnesses)
    if not normalised:
        raise ValueError("at least one witness is required")
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in normalised):
        raise ValueError("witness coordinates must be finite")
    if any(x * x + y * y > 1.0 + 1e-12 for x, y in normalised):
        raise ValueError("witnesses must lie in the unit disk")
    return normalised


def _master_objective(a: float, witnesses: tuple[Point, ...]) -> float:
    return max((abs(x) - a) ** 2 + y * y for x, y in witnesses)


def solve_finite_master(witnesses: Iterable[Point]) -> CoverSolution:
    """Solve the symmetric finite witness master exactly by breakpoint search.

    Each witness contributes ``a² - 2|x|a + x² + y²``.  The upper envelope
    can attain its minimum only at an endpoint, an active quadratic's vertex,
    or an intersection of two affine terms after the common ``a²`` is removed.
    """

    points = _normalise_witnesses(witnesses)
    terms = tuple((abs(x), x * x + y * y) for x, y in points)
    candidates = {0.0, 1.0}
    candidates.update(b for b, _ in terms if 0.0 <= b <= 1.0)
    for index, (first_b, first_d) in enumerate(terms):
        for second_b, second_d in terms[index + 1 :]:
            denominator = 2.0 * (first_b - second_b)
            if abs(denominator) <= 1e-15:
                continue
            crossing = (first_d - second_d) / denominator
            if 0.0 <= crossing <= 1.0:
                candidates.add(crossing)

    best_a = min(candidates, key=lambda a: (_master_objective(a, points), a))
    objective = _master_objective(best_a, points)
    return CoverSolution(a=best_a, radius=math.sqrt(objective), objective=objective)


def separate_power_distance(
    solution: CoverSolution,
    *,
    tolerance: float = 1e-9,
) -> SeparationResult:
    """Return an exact worst witness for squared power distance.

    On the unit-circle boundary the nearer-centre power distance is
    ``1 - 2a|x| + a² - r²``.  Its maximum is attained at either pole.
    """

    _validate_solution(solution)
    tolerance = _nonnegative_finite(tolerance, "tolerance")
    violation = 1.0 + solution.a * solution.a - solution.radius * solution.radius
    return SeparationResult(
        witness=(0.0, 1.0),
        violation=violation,
        is_violated=violation > tolerance,
    )


def verify_cover_independently(
    solution: CoverSolution,
    *,
    tolerance: float = 1e-9,
    boundary_samples: int = 4096,
) -> IndependentVerification:
    """Verify coverage on an independent deterministic boundary discretisation."""

    _validate_solution(solution)
    tolerance = _nonnegative_finite(tolerance, "tolerance")
    boundary_samples = _nonnegative_integer(boundary_samples, "boundary_samples")
    if boundary_samples < 4:
        raise ValueError("boundary_samples must be at least four")
    angles = {math.tau * index / boundary_samples for index in range(boundary_samples)}
    angles.update((0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0))
    maximum = -math.inf
    for angle in angles:
        x = math.cos(angle)
        y = math.sin(angle)
        left_power = (x + solution.a) ** 2 + y * y - solution.radius**2
        right_power = (x - solution.a) ** 2 + y * y - solution.radius**2
        maximum = max(maximum, min(left_power, right_power))
    return IndependentVerification(
        passed=maximum <= tolerance,
        maximum_power_violation=maximum,
        sample_count=len(angles),
    )


def run_constraint_generation(
    *,
    seed: int = 0,
    max_iterations: int = 4,
    tolerance: float = 1e-9,
    initial_witnesses: Iterable[Point] = DEFAULT_WITNESSES,
) -> ConstraintGenerationResult:
    """Run deterministic constraint generation with an explicit unresolved state."""

    seed = _nonnegative_integer(seed, "seed")
    max_iterations = _nonnegative_integer(max_iterations, "max_iterations")
    tolerance = _nonnegative_finite(tolerance, "tolerance")
    witnesses = _normalise_witnesses(initial_witnesses)
    added: list[Point] = []
    current_violations: list[float] = []
    best_seen: list[float] = []
    best_seen_violation = math.inf

    for augmentation_count in range(max_iterations + 1):
        solution = solve_finite_master((*witnesses, *added))
        separation = separate_power_distance(solution, tolerance=tolerance)
        current_violation = max(0.0, separation.violation)
        current_violations.append(current_violation)
        best_seen_violation = min(best_seen_violation, current_violation)
        best_seen.append(best_seen_violation)
        if not separation.is_violated:
            return ConstraintGenerationResult(
                solution=solution,
                converged=True,
                augmentations=augmentation_count,
                master_solves=augmentation_count + 1,
                oracle_calls=augmentation_count + 1,
                initial_witnesses=witnesses,
                added_witnesses=tuple(added),
                current_violations=tuple(current_violations),
                best_seen_positive_violations=tuple(best_seen),
                seed=seed,
                failure_reason=None,
            )
        if augmentation_count == max_iterations:
            break
        added.append(separation.witness)

    return ConstraintGenerationResult(
        solution=solution,
        converged=False,
        augmentations=max_iterations,
        master_solves=max_iterations + 1,
        oracle_calls=max_iterations + 1,
        initial_witnesses=witnesses,
        added_witnesses=tuple(added),
        current_violations=tuple(current_violations),
        best_seen_positive_violations=tuple(best_seen),
        seed=seed,
        failure_reason="maximum constraint-generation iterations reached",
    )
