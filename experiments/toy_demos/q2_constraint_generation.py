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

    ``violation_upper_bounds[k]`` is the smallest nonnegative exact residual
    observed through iteration ``k``.  It is therefore an attainable upper
    bound on the best residual among evaluated iterates and is non-increasing;
    it is deliberately not labelled as the current iterate's violation.
    """

    solution: CoverSolution
    converged: bool
    iterations: int
    initial_witnesses: tuple[Point, ...]
    added_witnesses: tuple[Point, ...]
    violation_upper_bounds: tuple[float, ...]
    seed: int
    failure_reason: str | None


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

    if boundary_samples < 4:
        raise ValueError("boundary_samples must be at least four")
    maximum = -math.inf
    for index in range(boundary_samples):
        angle = math.tau * index / boundary_samples
        x = math.cos(angle)
        y = math.sin(angle)
        left_power = (x + solution.a) ** 2 + y * y - solution.radius**2
        right_power = (x - solution.a) ** 2 + y * y - solution.radius**2
        maximum = max(maximum, min(left_power, right_power))
    return IndependentVerification(
        passed=maximum <= tolerance,
        maximum_power_violation=maximum,
        sample_count=boundary_samples,
    )


def run_constraint_generation(
    *,
    seed: int = 0,
    max_iterations: int = 4,
    tolerance: float = 1e-9,
    initial_witnesses: Iterable[Point] = DEFAULT_WITNESSES,
) -> ConstraintGenerationResult:
    """Run deterministic constraint generation with an explicit unresolved state."""

    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    witnesses = _normalise_witnesses(initial_witnesses)
    added: list[Point] = []
    upper_bounds: list[float] = []
    incumbent_upper_bound = math.inf
    solution = solve_finite_master(witnesses)

    for augmentation_count in range(max_iterations + 1):
        solution = solve_finite_master((*witnesses, *added))
        separation = separate_power_distance(solution, tolerance=tolerance)
        incumbent_upper_bound = min(incumbent_upper_bound, max(0.0, separation.violation))
        upper_bounds.append(incumbent_upper_bound)
        if not separation.is_violated:
            return ConstraintGenerationResult(
                solution=solution,
                converged=True,
                iterations=augmentation_count,
                initial_witnesses=witnesses,
                added_witnesses=tuple(added),
                violation_upper_bounds=tuple(upper_bounds),
                seed=seed,
                failure_reason=None,
            )
        if augmentation_count == max_iterations:
            break
        added.append(separation.witness)

    return ConstraintGenerationResult(
        solution=solution,
        converged=False,
        iterations=max_iterations,
        initial_witnesses=witnesses,
        added_witnesses=tuple(added),
        violation_upper_bounds=tuple(upper_bounds),
        seed=seed,
        failure_reason="maximum constraint-generation iterations reached",
    )
