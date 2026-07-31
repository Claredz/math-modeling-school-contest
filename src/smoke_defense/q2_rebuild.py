"""Q2 event candidates and conservative continuous joint-coverage verifier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from math import isclose

import numpy as np
from scipy.optimize import brentq, minimize

from smoke_defense.coverage import Disk, certify_union_coverage
from smoke_defense.events import ClosedInterval
from smoke_defense.path_constraints import certify_operation_radius
from smoke_defense.paths import UAV_SPEED_MPS, LinearFlightSegment, ShipborneUavPath
from smoke_defense.q1_rebuild import (
    BURST_DELAY_S,
    RESPONSE_DELAY_S,
    Q1Problem,
)
from smoke_defense.smoke import SmokeCloud


class Q2CertificationStatus(StrEnum):
    CERTIFIED_FEASIBLE = "certified_feasible"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Q2Bomb:
    command_time_s: float
    drop_time_s: float
    burst_time_s: float
    drop_position_m: np.ndarray
    burst_center_m: np.ndarray
    smoke: SmokeCloud


@dataclass(frozen=True)
class Q2Plan:
    bombs: tuple[Q2Bomb, ...]
    path: ShipborneUavPath
    solver_native_success: bool | None = None


@dataclass(frozen=True)
class Q2JointCertificate:
    status: Q2CertificationStatus
    coverage_lower_s: float
    coverage_upper_s: float
    total_exposure_lower_s: float
    total_exposure_upper_s: float
    maximum_continuous_exposure_s: float
    joint_gain_s: float
    witness_time_s: float | None = None
    unresolved_intervals: tuple[ClosedInterval, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class Q2CandidateResult:
    """One event-parameterized plan and its independent joint certificate."""

    burst_times_s: tuple[float, ...]
    certificate: Q2JointCertificate
    center_times_s: tuple[float, ...] = ()
    solver_native_success: bool | None = None


@dataclass(frozen=True)
class Q2SolveResult:
    """Best known Q2 plan; this is deliberately not a global-optimum claim."""

    best: Q2CandidateResult
    candidates: tuple[Q2CandidateResult, ...]
    warm_start_count: int
    polish_success: bool
    polish_status: str


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.array([1.0, 0.0])
    return vector / norm


def _connect_segment(
    start_time_s: float,
    start_position_m: np.ndarray,
    end_time_s: float,
    end_position_m: np.ndarray,
    final_heading: np.ndarray,
) -> tuple[LinearFlightSegment, ...]:
    if end_time_s <= start_time_s:
        raise ValueError("Q2 drop times must be strictly increasing")
    start = np.asarray(start_position_m, dtype=float)
    end = np.asarray(end_position_m, dtype=float)
    length = UAV_SPEED_MPS * (end_time_s - start_time_s)
    displacement = float(np.linalg.norm(end - start))
    if displacement > length + 1e-8:
        raise ValueError("Q2 release point is unreachable")
    heading = _unit(np.asarray(final_heading, dtype=float))
    if isclose(displacement, length, abs_tol=1e-8) and np.allclose(
        _unit(end - start), heading, rtol=0.0, atol=1e-8
    ):
        return (LinearFlightSegment(start_time_s, end_time_s, start, end),)

    relative = end - start

    def residual(final_leg_m: float) -> float:
        waypoint_relative = relative - final_leg_m * heading
        return float(np.linalg.norm(waypoint_relative) + final_leg_m - length)

    final_leg = float(brentq(residual, 0.0, length, xtol=1e-11))
    waypoint = end - final_leg * heading
    first_length = float(np.linalg.norm(waypoint - start))
    if first_length <= 1e-8 or final_leg <= 1e-8:
        raise ValueError("Q2 path requires two positive flight legs")
    turn_time = start_time_s + first_length / UAV_SPEED_MPS
    return (
        LinearFlightSegment(start_time_s, turn_time, start, waypoint),
        LinearFlightSegment(turn_time, end_time_s, waypoint, end),
    )


def construct_q2_plan(
    problem: Q1Problem,
    *,
    burst_times_s: tuple[float, ...],
    center_times_s: tuple[float, ...] | None = None,
) -> Q2Plan:
    if not 1 <= len(burst_times_s) <= 3:
        raise ValueError("Q2 uses between one and three bombs")
    sorted_bursts = tuple(float(value) for value in burst_times_s)
    if tuple(sorted(sorted_bursts)) != sorted_bursts:
        raise ValueError("Q2 burst times must be sorted")
    if center_times_s is None:
        sorted_centers = sorted_bursts
    else:
        if len(center_times_s) != len(sorted_bursts):
            raise ValueError("Q2 center and burst counts must match")
        sorted_centers = tuple(float(value) for value in center_times_s)
    segments: list[LinearFlightSegment] = []
    bombs: list[Q2Bomb] = []
    previous_time = 0.0
    previous_position = problem.ship.position(0.0)
    for burst_time, center_time in zip(sorted_bursts, sorted_centers, strict=True):
        drop_time = burst_time - BURST_DELAY_S
        command_time = drop_time - RESPONSE_DELAY_S
        if command_time < 0:
            raise ValueError("negative Q2 command time is forbidden")
        center = problem.ship.position(center_time)
        heading = _unit(center - previous_position)
        drop_position = center - UAV_SPEED_MPS * BURST_DELAY_S * heading
        segments.extend(
            _connect_segment(
                previous_time,
                previous_position,
                drop_time,
                drop_position,
                heading,
            )
        )
        bombs.append(
            Q2Bomb(
                command_time_s=command_time,
                drop_time_s=drop_time,
                burst_time_s=burst_time,
                drop_position_m=drop_position,
                burst_center_m=center,
                smoke=SmokeCloud(
                    burst_time_s=burst_time,
                    burst_center_m=center,
                    maximum_radius_m=problem.constants.smoke.maximum_radius_m,
                    hold_duration_s=problem.constants.smoke.hold_duration_s,
                    decay_duration_s=problem.constants.smoke.decay_duration_s,
                ),
            )
        )
        previous_time = drop_time
        previous_position = drop_position
    path = ShipborneUavPath(
        ship=problem.ship,
        takeoff_time_s=0.0,
        segments=tuple(segments),
    )
    if any(
        second.drop_time_s - first.drop_time_s < 1.0 - 1e-9
        for first, second in zip(bombs[:-1], bombs[1:], strict=True)
    ):
        raise ValueError("same-UAV drop spacing must be at least one second")
    return Q2Plan(tuple(bombs), path)


def _active_disks(
    smokes: tuple[SmokeCloud, ...], time_s: float, shrink_s: float = 0.0
) -> tuple[Disk, ...]:
    disks = []
    for smoke in smokes:
        radius = max(0.0, smoke.radius(time_s) - smoke.decay_rate_mps * shrink_s)
        if radius > 0:
            disks.append(Disk(smoke.center(time_s), radius))
    return tuple(disks)


def certify_joint_coverage(
    *,
    ship_position,
    detection_components: tuple[ClosedInterval, ...],
    smokes: tuple[SmokeCloud, ...],
    ship_radius_m: float = 80.0,
    ship_speed_bound_mps: float = 7.71,
    initial_polygon_sides: int = 32,
    maximum_polygon_sides: int = 2048,
    time_tolerance_s: float = 1e-3,
) -> Q2JointCertificate:
    """Certify by time subdivision with a conservative spatial envelope."""

    if not detection_components:
        return Q2JointCertificate(
            Q2CertificationStatus.CERTIFIED_FEASIBLE,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    lower = 0.0
    upper = 0.0
    unresolved: list[ClosedInterval] = []
    total = sum(component.duration_s for component in detection_components)
    maximum_exposure = 0.0
    for component in detection_components:
        breakpoints = {component.start_s, component.end_s}
        for smoke in smokes:
            for event_time in (smoke.burst_time_s, smoke.hold_end_time_s, smoke.failure_time_s):
                if component.start_s <= event_time <= component.end_s:
                    breakpoints.add(event_time)
        stack = list(zip(sorted(breakpoints)[:-1], sorted(breakpoints)[1:], strict=True))
        while stack:
            left, right = stack.pop()
            if right <= left:
                continue
            half = 0.5 * (right - left)
            midpoint = 0.5 * (left + right)
            target = Disk(
                np.asarray(ship_position(midpoint), dtype=float),
                ship_radius_m + ship_speed_bound_mps * half,
            )
            certificate = certify_union_coverage(
                target,
                _active_disks(smokes, midpoint, shrink_s=half),
                initial_polygon_sides=initial_polygon_sides,
                maximum_polygon_sides=maximum_polygon_sides,
            )
            if certificate.status.value == "certified_feasible":
                lower += right - left
                upper += right - left
                maximum_exposure = max(maximum_exposure, 0.0)
                continue
            exact_target = Disk(np.asarray(ship_position(midpoint), dtype=float), ship_radius_m)
            exact_certificate = certify_union_coverage(
                exact_target,
                _active_disks(smokes, midpoint),
                initial_polygon_sides=initial_polygon_sides,
                maximum_polygon_sides=maximum_polygon_sides,
            )
            if exact_certificate.status.value == "certified_infeasible":
                return Q2JointCertificate(
                    Q2CertificationStatus.CERTIFIED_INFEASIBLE,
                    lower,
                    upper,
                    total - upper,
                    total - lower,
                    total - lower,
                    upper,
                    witness_time_s=midpoint,
                    unresolved_intervals=tuple(unresolved),
                    reason=exact_certificate.reason,
                )
            if right - left <= time_tolerance_s:
                unresolved.append(ClosedInterval(left, right))
                upper += right - left
                continue
            stack.extend(((left, midpoint), (midpoint, right)))
    status = (
        Q2CertificationStatus.UNRESOLVED
        if unresolved
        else Q2CertificationStatus.CERTIFIED_FEASIBLE
    )
    return Q2JointCertificate(
        status,
        lower,
        upper,
        total - upper,
        total - lower,
        total - lower,
        max(0.0, lower),
        unresolved_intervals=tuple(unresolved),
        reason=(
            "time-space envelope closed"
            if not unresolved
            else "time tolerance left unresolved cells"
        ),
    )


def verify_q2_plan(problem: Q1Problem, plan: Q2Plan) -> Q2JointCertificate:
    radius = certify_operation_radius(
        plan.path,
        operation_radius_m=problem.constants.uav.operation_radius_m,
    )
    if radius.status != "certified_feasible":
        return Q2JointCertificate(
            Q2CertificationStatus.CERTIFIED_INFEASIBLE,
            0.0,
            0.0,
            sum(item.duration_s for item in problem.detection.components),
            sum(item.duration_s for item in problem.detection.components),
            sum(item.duration_s for item in problem.detection.components),
            0.0,
            reason=radius.reason,
        )
    return certify_joint_coverage(
        ship_position=problem.ship.position,
        detection_components=problem.detection.components,
        smokes=tuple(bomb.smoke for bomb in plan.bombs),
        ship_radius_m=problem.constants.ship.effective_radius_m,
        ship_speed_bound_mps=problem.constants.ship.speed_mps,
    )


def q2_verification_rank(result: Q2JointCertificate) -> tuple[float, ...]:
    """Strict feasibility-first ranking for candidate comparison."""

    status_rank = {
        Q2CertificationStatus.CERTIFIED_FEASIBLE: 2.0,
        Q2CertificationStatus.UNRESOLVED: 1.0,
        Q2CertificationStatus.CERTIFIED_INFEASIBLE: 0.0,
    }[result.status]
    return (
        status_rank,
        round(result.coverage_lower_s, 9),
        -round(result.maximum_continuous_exposure_s, 9),
        round(result.joint_gain_s, 9),
        -round(result.coverage_upper_s - result.coverage_lower_s, 9),
    )


def _q2_time_grid(problem: Q1Problem, warm_burst_times_s: tuple[float, ...]) -> tuple[float, ...]:
    components = problem.detection.components
    start = max(5.5, components[0].start_s)
    end = components[-1].end_s
    anchors = [start, end, *warm_burst_times_s]
    anchors.extend(np.linspace(start, end, num=9).tolist())
    for component in components:
        anchors.extend(
            (
                component.start_s,
                component.end_s,
                0.5 * (component.start_s + component.end_s),
            )
        )
    grid = sorted({round(float(np.clip(value, start, end)), 6) for value in anchors})
    return tuple(value for value in grid if value >= 5.5)


def _candidate_burst_times(
    problem: Q1Problem,
    *,
    warm_burst_times_s: tuple[float, ...] = (),
    maximum_candidates: int = 28,
) -> tuple[tuple[float, ...], ...]:
    grid = _q2_time_grid(problem, warm_burst_times_s)
    candidates: list[tuple[float, ...]] = []
    for count in (1, 2, 3):
        candidates.extend(combinations(grid, count))
    warm = tuple(sorted(float(value) for value in warm_burst_times_s))
    ordered: list[tuple[float, ...]] = []
    if warm and all(value in grid for value in warm):
        ordered.append(warm[:3])
    ordered.extend(candidates)
    unique = list(dict.fromkeys(ordered))
    return tuple(unique[:maximum_candidates])


def solve_q2_candidates(
    problem: Q1Problem,
    *,
    warm_burst_times_s: tuple[float, ...] = (),
    warm_center_times_s: tuple[float, ...] = (),
    maximum_candidates: int = 28,
    polish: bool = True,
) -> Q2SolveResult:
    """Generate warm-start events, locally refine, and verify every survivor.

    The returned plan is the best known candidate under a bounded search.  It
    is never labelled as a globally exact optimum.
    """

    burst_candidates = _candidate_burst_times(
        problem,
        warm_burst_times_s=warm_burst_times_s,
        maximum_candidates=maximum_candidates,
    )
    components = problem.detection.components
    center_start = max(5.5, components[0].start_s)
    center_end = components[-1].end_s
    if len(warm_burst_times_s) == len(warm_center_times_s) and warm_burst_times_s:
        center_offset = float(
            np.mean(
                np.asarray(warm_center_times_s, dtype=float)
                - np.asarray(warm_burst_times_s, dtype=float)
            )
        )
    else:
        center_offset = 0.0

    def centers_for(bursts: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            float(np.clip(value + center_offset, center_start, center_end))
            for value in bursts
        )

    evaluations: list[Q2CandidateResult] = []
    for bursts in burst_candidates:
        centers = centers_for(bursts)
        try:
            certificate = verify_q2_plan(
                problem,
                construct_q2_plan(
                    problem, burst_times_s=bursts, center_times_s=centers
                ),
            )
        except (ValueError, RuntimeError) as exc:
            certificate = Q2JointCertificate(
                Q2CertificationStatus.CERTIFIED_INFEASIBLE,
                0.0,
                0.0,
                sum(item.duration_s for item in problem.detection.components),
                sum(item.duration_s for item in problem.detection.components),
                0.0,
                0.0,
                reason=str(exc),
            )
        evaluations.append(Q2CandidateResult(bursts, certificate, centers))
    if not evaluations:
        raise ValueError("Q2 candidate generator produced no candidates")
    best = max(evaluations, key=lambda item: q2_verification_rank(item.certificate))
    polish_success = False
    polish_status = "not_started"
    if polish and len(best.burst_times_s) >= 1:
        count = len(best.burst_times_s)
        start = max(5.5, problem.detection.components[0].start_s)
        end = problem.detection.components[-1].end_s

        def objective(vector: np.ndarray) -> float:
            values = tuple(float(value) for value in vector)
            if any(
                second <= first + 1.0
                for first, second in zip(values[:-1], values[1:], strict=True)
            ):
                return 1e9
            try:
                certificate = verify_q2_plan(
                    problem,
                    construct_q2_plan(
                        problem,
                        burst_times_s=values,
                        center_times_s=centers_for(values),
                    ),
                )
            except (ValueError, RuntimeError):
                return 1e9
            rank = q2_verification_rank(certificate)
            return -(rank[0] * 1e6 + rank[1] * 1e3 + rank[2] + rank[3] * 1e-2)

        result = minimize(
            objective,
            np.asarray(best.burst_times_s, dtype=float),
            method="SLSQP",
            bounds=[(start, end)] * count,
            constraints=(
                ({"type": "ineq", "fun": lambda vector: np.diff(vector) - 1.0},)
                if count > 1
                else ()
            ),
            options={"maxiter": 8, "ftol": 1e-5, "disp": False},
        )
        polish_success = bool(result.success)
        polish_status = str(result.message)
        if result.success:
            refined = tuple(float(value) for value in result.x)
            try:
                refined_certificate = verify_q2_plan(
                    problem,
                    construct_q2_plan(
                        problem,
                        burst_times_s=refined,
                        center_times_s=centers_for(refined),
                    ),
                )
                refined_result = Q2CandidateResult(
                    refined,
                    refined_certificate,
                    centers_for(refined),
                    solver_native_success=True,
                )
                evaluations.append(refined_result)
                if q2_verification_rank(refined_certificate) > q2_verification_rank(
                    best.certificate
                ):
                    best = refined_result
            except (ValueError, RuntimeError):
                polish_status = "polish candidate rejected by verifier"
    return Q2SolveResult(
        best=best,
        candidates=tuple(evaluations),
        warm_start_count=len(warm_burst_times_s),
        polish_success=polish_success,
        polish_status=polish_status,
    )
