"""Stage 5 Q1 rebuild with an explicit solver/verifier boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import differential_evolution, minimize, shgo
from scipy.stats import qmc

from smoke_defense.candidates import (
    build_shipborne_release_path,
    evaluate_smoke_against_detection,
)
from smoke_defense.constants import ProblemConstants, load_problem_constants
from smoke_defense.coverage import CertificationStatus
from smoke_defense.detection import DetectionSet
from smoke_defense.dynamics import ShipMotion
from smoke_defense.path_constraints import certify_operation_radius
from smoke_defense.paths import ShipborneUavPath
from smoke_defense.q1 import _scenario_objects, solve_q1_scenario
from smoke_defense.scenario import Scenario
from smoke_defense.smoke import SmokeCloud, detonation_position
from smoke_defense.verification import certify_single_smoke_continuous_coverage

RESPONSE_DELAY_S = 2.0
BURST_DELAY_S = 3.5
Q1_METHODS = (
    "multistart_slsqp",
    "sobol_slsqp",
    "shgo",
    "differential_evolution_slsqp",
)


@dataclass(frozen=True)
class Q1Problem:
    """Immutable formal scenario context shared by all candidate evaluations."""

    scenario: Scenario
    constants: ProblemConstants
    ship: ShipMotion
    detection: DetectionSet


@dataclass(frozen=True)
class Q1CandidateDecision:
    """Solver output only; certification deliberately lives elsewhere."""

    command_time_s: float
    drop_time_s: float
    burst_time_s: float
    center_time_s: float
    drop_position_m: np.ndarray
    path: ShipborneUavPath
    smoke: SmokeCloud
    flight_distance_m: float

    def with_smoke_center(self, center_m: np.ndarray) -> Q1CandidateDecision:
        """Return a tampered copy used by independent-verifier tests."""

        smoke = replace(
            self.smoke,
            burst_center_m=np.asarray(center_m, dtype=float),
        )
        return replace(self, smoke=smoke)


@dataclass(frozen=True)
class Q1Verification:
    """Independent continuous verification outcome for one solver candidate."""

    status: str
    covered_duration_s: float
    exposed_duration_s: float
    maximum_exposure_s: float
    minimum_margin_m: float
    flight_distance_m: float
    reason: str
    witness_time_s: float | None = None
    solver_native_success: bool | None = None


@dataclass(frozen=True)
class Q1MethodResult:
    method: str
    seed: int
    evaluation_budget: int
    evaluations: int
    bounds: tuple[tuple[float, float], tuple[float, float]]
    native_success: bool
    native_status: str
    best_candidate: Q1CandidateDecision | None
    verification: Q1Verification | None


def build_q1_problem(
    scenario: Scenario,
    constants: ProblemConstants | None = None,
) -> Q1Problem:
    """Build the formal IPP context once, outside optimizer evaluations."""

    if scenario.model_layer != "formal_baseline":
        raise ValueError("Q1 rebuild requires a formal_baseline scenario")
    constants = constants or load_problem_constants()
    result = solve_q1_scenario(scenario, constants)
    ship, _missile, _appearance = _scenario_objects(scenario, constants)
    return Q1Problem(
        scenario=scenario,
        constants=constants,
        ship=ship,
        detection=result.detection,
    )


def construct_q1_candidate(
    problem: Q1Problem,
    *,
    burst_time_s: float,
    center_time_s: float,
) -> Q1CandidateDecision:
    """Construct a causal piecewise-linear UAV and fixed-centre smoke plan."""

    burst_time_s = float(burst_time_s)
    center_time_s = float(center_time_s)
    drop_time_s = burst_time_s - BURST_DELAY_S
    command_time_s = drop_time_s - RESPONSE_DELAY_S
    if command_time_s < 0:
        raise ValueError("negative command time is forbidden in the formal baseline")
    burst_center = problem.ship.position(center_time_s)
    path, drop_position, _burst_heading = build_shipborne_release_path(
        ship=problem.ship,
        takeoff_time_s=0.0,
        release_time_s=drop_time_s,
        burst_center_m=burst_center,
        detonation_delay_s=BURST_DELAY_S,
    )
    smoke = SmokeCloud(
        burst_time_s=burst_time_s,
        burst_center_m=burst_center,
        maximum_radius_m=problem.constants.smoke.maximum_radius_m,
        hold_duration_s=problem.constants.smoke.hold_duration_s,
        decay_duration_s=problem.constants.smoke.decay_duration_s,
    )
    flight_distance = sum(
        float(np.linalg.norm(segment.end_position_m - segment.start_position_m))
        for segment in path.segments
    )
    return Q1CandidateDecision(
        command_time_s=command_time_s,
        drop_time_s=drop_time_s,
        burst_time_s=burst_time_s,
        center_time_s=center_time_s,
        drop_position_m=drop_position,
        path=path,
        smoke=smoke,
        flight_distance_m=flight_distance,
    )


def _infeasible_verification(
    candidate: Q1CandidateDecision,
    reason: str,
) -> Q1Verification:
    return Q1Verification(
        status="certified_infeasible",
        covered_duration_s=0.0,
        exposed_duration_s=0.0,
        maximum_exposure_s=0.0,
        minimum_margin_m=float("-inf"),
        flight_distance_m=candidate.flight_distance_m,
        reason=reason,
    )


def verify_q1_candidate(
    problem: Q1Problem,
    candidate: Q1CandidateDecision,
) -> Q1Verification:
    """Recompute event, path and continuous coverage facts independently."""

    if candidate.command_time_s < 0:
        return _infeasible_verification(candidate, "negative command time")
    if not np.isclose(
        candidate.drop_time_s - candidate.command_time_s,
        RESPONSE_DELAY_S,
        rtol=0.0,
        atol=1e-10,
    ):
        return _infeasible_verification(candidate, "command/drop delay mismatch")
    if not np.isclose(
        candidate.burst_time_s - candidate.drop_time_s,
        BURST_DELAY_S,
        rtol=0.0,
        atol=1e-10,
    ):
        return _infeasible_verification(candidate, "drop/burst delay mismatch")

    drop_position = candidate.path.position(candidate.drop_time_s)
    burst_position = detonation_position(
        drop_position,
        candidate.path.velocity(candidate.drop_time_s),
        delay_s=BURST_DELAY_S,
    )
    if not np.allclose(
        burst_position,
        candidate.smoke.burst_center_m,
        rtol=0.0,
        atol=1e-7,
    ):
        return _infeasible_verification(
            candidate,
            "path and burst centre are inconsistent",
        )
    radius_certificate = certify_operation_radius(
        candidate.path,
        operation_radius_m=problem.constants.uav.operation_radius_m,
    )
    if radius_certificate.status != "certified_feasible":
        return _infeasible_verification(
            candidate,
            f"path constraint: {radius_certificate.reason}",
        )

    (
        _covered,
        covered_duration,
        exposed_duration,
        maximum_exposure,
        minimum_margin,
        _legacy_status,
    ) = evaluate_smoke_against_detection(
        ship=problem.ship,
        smoke=candidate.smoke,
        detection=problem.detection,
        ship_radius_m=problem.constants.ship.effective_radius_m,
    )
    certificate = certify_single_smoke_continuous_coverage(
        ship_position=problem.ship.position,
        smoke=candidate.smoke,
        detection_components=problem.detection.components,
        ship_radius_m=problem.constants.ship.effective_radius_m,
        ship_speed_bound_mps=problem.constants.ship.speed_mps,
    )
    status = {
        CertificationStatus.CERTIFIED_FEASIBLE: "certified_feasible",
        CertificationStatus.CERTIFIED_INFEASIBLE: "certified_infeasible",
        CertificationStatus.INDETERMINATE: "unresolved",
    }[certificate.status]
    return Q1Verification(
        status=status,
        covered_duration_s=covered_duration,
        exposed_duration_s=exposed_duration,
        maximum_exposure_s=maximum_exposure,
        minimum_margin_m=minimum_margin,
        flight_distance_m=candidate.flight_distance_m,
        reason=certificate.reason,
        witness_time_s=certificate.witness_time_s,
    )


def q1_verification_rank(result: Q1Verification) -> tuple[float, ...]:
    """Approved strict-coverage-first lexicographic objective hierarchy."""

    status_rank = {
        "certified_feasible": 2.0,
        "unresolved": 1.0,
        "certified_infeasible": 0.0,
    }[result.status]
    return (
        status_rank,
        round(result.covered_duration_s, 9),
        -result.maximum_exposure_s,
        result.minimum_margin_m,
        -result.flight_distance_m,
    )


def _q1_bounds(problem: Q1Problem) -> tuple[tuple[float, float], tuple[float, float]]:
    start = max(5.5, problem.detection.components[0].start_s)
    end = max(start + 1e-3, problem.detection.components[-1].end_s)
    return ((start, end), (start, end))


def _scalar_objective(result: Q1Verification) -> float:
    status = {
        "certified_feasible": 2.0,
        "unresolved": 1.0,
        "certified_infeasible": 0.0,
    }[result.status]
    margin = result.minimum_margin_m if np.isfinite(result.minimum_margin_m) else -1e6
    return -(
        status * 1e6
        + result.covered_duration_s * 1e3
        - result.maximum_exposure_s * 10.0
        + margin
        - result.flight_distance_m * 1e-3
    )


def _run_method(
    problem: Q1Problem,
    *,
    method: str,
    seed: int,
    evaluation_budget: int,
) -> Q1MethodResult:
    if method not in Q1_METHODS:
        raise ValueError(f"unknown Q1 method: {method}")
    if evaluation_budget < 4:
        raise ValueError("Q1 evaluation budget must be at least 4")
    bounds = _q1_bounds(problem)
    rng = np.random.default_rng(seed)
    cache: dict[tuple[float, float], tuple[Q1CandidateDecision | None, Q1Verification]] = {}
    native_success = False
    native_status = "not_started"

    def evaluate(vector: np.ndarray) -> float:
        nonlocal native_status
        key = tuple(np.round(np.asarray(vector, dtype=float), 10))
        if key not in cache:
            if len(cache) >= evaluation_budget:
                native_status = "evaluation_budget_exhausted"
                return 1e12
            try:
                candidate = construct_q1_candidate(
                    problem,
                    burst_time_s=float(key[0]),
                    center_time_s=float(key[1]),
                )
                verification = verify_q1_candidate(problem, candidate)
            except (ValueError, RuntimeError) as exc:
                candidate = None
                verification = Q1Verification(
                    status="certified_infeasible",
                    covered_duration_s=0.0,
                    exposed_duration_s=0.0,
                    maximum_exposure_s=0.0,
                    minimum_margin_m=-1e6,
                    flight_distance_m=0.0,
                    reason=str(exc),
                )
            cache[key] = (candidate, verification)
        return _scalar_objective(cache[key][1])

    def local(start: np.ndarray) -> None:
        nonlocal native_success, native_status
        result = minimize(
            evaluate,
            np.asarray(start, dtype=float),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": max(2, evaluation_budget // 4), "ftol": 1e-8},
        )
        native_success = native_success or bool(result.success)
        native_status = str(result.message)

    try:
        midpoint = np.mean(np.asarray(bounds), axis=1)
        if method == "multistart_slsqp":
            starts = [midpoint]
            starts.extend(rng.uniform(*bounds[0], size=2).tolist() for _ in range(3))
            for start in starts:
                if len(cache) >= evaluation_budget:
                    break
                local(np.asarray(start))
        elif method == "sobol_slsqp":
            sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
            points = qmc.scale(
                sampler.random_base2(m=3),
                np.asarray(bounds)[:, 0],
                np.asarray(bounds)[:, 1],
            )
            for point in points:
                evaluate(point)
                if len(cache) >= evaluation_budget:
                    break
            if cache:
                local(np.asarray(min(cache, key=lambda key: cache[key][1].covered_duration_s)))
        elif method == "shgo":
            result = shgo(
                evaluate,
                bounds,
                n=min(8, evaluation_budget),
                iters=1,
                options={"minimize_every_iter": False},
            )
            native_success = bool(result.success)
            native_status = str(result.message)
        else:
            result = differential_evolution(
                evaluate,
                bounds,
                seed=seed,
                maxiter=1,
                popsize=max(4, evaluation_budget // 8),
                polish=False,
                updating="immediate",
            )
            native_success = bool(result.success)
            native_status = str(result.message)
            if len(cache) < evaluation_budget:
                local(np.asarray(result.x))
    except (ValueError, RuntimeError) as exc:
        native_status = str(exc)

    if not cache:
        return Q1MethodResult(
            method,
            seed,
            evaluation_budget,
            0,
            bounds,
            native_success,
            native_status,
            None,
            None,
        )
    key = max(cache, key=lambda item: q1_verification_rank(cache[item][1]))
    candidate, verification = cache[key]
    if verification.solver_native_success is None:
        verification = replace(verification, solver_native_success=native_success)
    return Q1MethodResult(
        method=method,
        seed=seed,
        evaluation_budget=evaluation_budget,
        evaluations=len(cache),
        bounds=bounds,
        native_success=native_success,
        native_status=native_status,
        best_candidate=candidate,
        verification=verification,
    )


def benchmark_q1_methods(
    problem: Q1Problem,
    *,
    seed: int = 20260731,
    evaluation_budget: int = 48,
) -> tuple[Q1MethodResult, ...]:
    """Run the approved routes under a common seed, bounds, budget and verifier."""

    return tuple(
        _run_method(
            problem,
            method=method,
            seed=seed,
            evaluation_budget=evaluation_budget,
        )
        for method in Q1_METHODS
    )
