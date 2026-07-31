"""Q3 cooperative three-UAV, one-bomb-per-UAV reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from smoke_defense.path_constraints import (
    certify_operation_radius,
    certify_pairwise_separation,
)
from smoke_defense.q1_rebuild import Q1Problem
from smoke_defense.q2_rebuild import (
    Q2CertificationStatus,
    Q2JointCertificate,
    certify_joint_coverage,
    construct_q2_plan,
)


@dataclass(frozen=True)
class Q3Plan:
    burst_times_s: tuple[float, ...]
    center_times_s: tuple[float, ...]
    paths: tuple[object, ...]
    smokes: tuple[object, ...]
    interpretation: str = "exactly_one_bomb_per_uav"


@dataclass(frozen=True)
class Q3Certificate:
    status: Q2CertificationStatus
    joint: Q2JointCertificate
    operation_radius_ok: bool
    pairwise_conflict_ok: bool
    minimum_pairwise_distance_m: float
    epsilon_constraints: dict[str, float]
    interpretation: str
    reason: str = ""


def construct_q3_plan(
    problem: Q1Problem,
    *,
    burst_times_s: tuple[float, float, float],
    center_times_s: tuple[float, float, float] | None = None,
) -> Q3Plan:
    if len(burst_times_s) != 3:
        raise ValueError("Q3 main interpretation requires three bombs")
    centers = center_times_s or burst_times_s
    if len(centers) != 3:
        raise ValueError("Q3 burst and centre counts must match")
    single_plans = tuple(
        construct_q2_plan(
            problem,
            burst_times_s=(float(burst),),
            center_times_s=(float(center),),
        )
        for burst, center in zip(burst_times_s, centers, strict=True)
    )
    return Q3Plan(
        burst_times_s=tuple(float(value) for value in burst_times_s),
        center_times_s=tuple(float(value) for value in centers),
        paths=tuple(plan.path for plan in single_plans),
        smokes=tuple(plan.bombs[0].smoke for plan in single_plans),
    )


def verify_q3_plan(problem: Q1Problem, plan: Q3Plan) -> Q3Certificate:
    radius_results = tuple(
        certify_operation_radius(
            path,
            operation_radius_m=problem.constants.uav.operation_radius_m,
        )
        for path in plan.paths
    )
    if any(item.status != "certified_feasible" for item in radius_results):
        joint = Q2JointCertificate(
            Q2CertificationStatus.CERTIFIED_INFEASIBLE,
            0.0,
            0.0,
            sum(item.duration_s for item in problem.detection.components),
            sum(item.duration_s for item in problem.detection.components),
            0.0,
            0.0,
            reason="at least one UAV exceeded the operation radius",
        )
        return Q3Certificate(
            joint.status,
            joint,
            False,
            False,
            0.0,
            {"max_continuous_exposure_s": joint.maximum_continuous_exposure_s},
            plan.interpretation,
            joint.reason,
        )
    pairwise_results = tuple(
        certify_pairwise_separation(
            first,
            second,
            safe_distance_m=0.0,
        )
        for index, first in enumerate(plan.paths)
        for second in plan.paths[index + 1 :]
    )
    pairwise_conflict_ok = all(
        item.status == "certified_feasible" for item in pairwise_results
    )
    minimum_pairwise_distance = min(
        (item.minimum_value or 0.0 for item in pairwise_results),
        default=0.0,
    )
    joint = certify_joint_coverage(
        ship_position=problem.ship.position,
        detection_components=problem.detection.components,
        smokes=plan.smokes,
        ship_radius_m=problem.constants.ship.effective_radius_m,
        ship_speed_bound_mps=problem.constants.ship.speed_mps,
    )
    return Q3Certificate(
        joint.status,
        joint,
        True,
        pairwise_conflict_ok,
        minimum_pairwise_distance,
        {
            "max_continuous_exposure_s": joint.maximum_continuous_exposure_s,
            "minimum_pairwise_distance_m": minimum_pairwise_distance,
        },
        plan.interpretation,
        joint.reason,
    )


def q3_verification_rank(certificate: Q3Certificate) -> tuple[float, ...]:
    """Success-first lexicographic rank with explicit epsilon constraints."""

    status_rank = {
        Q2CertificationStatus.CERTIFIED_FEASIBLE: 2.0,
        Q2CertificationStatus.UNRESOLVED: 1.0,
        Q2CertificationStatus.CERTIFIED_INFEASIBLE: 0.0,
    }[certificate.status]
    return (
        status_rank,
        float(certificate.pairwise_conflict_ok),
        certificate.joint.coverage_lower_s,
        -certificate.joint.maximum_continuous_exposure_s,
        certificate.minimum_pairwise_distance_m,
    )


def generate_q3_plan(
    problem: Q1Problem,
    *,
    warm_burst_times_s: tuple[float, ...] = (),
    warm_center_times_s: tuple[float, ...] = (),
) -> tuple[Q3Plan, Q3Certificate]:
    start = max(5.5, problem.detection.components[0].start_s)
    end = problem.detection.components[-1].end_s
    span = end - start
    if len(warm_burst_times_s) >= 2:
        bursts = tuple(sorted((*warm_burst_times_s[:2], end - 1e-3)))
        if len(warm_center_times_s) >= 2:
            centers = tuple(
                sorted((*warm_center_times_s[:2], min(end, end - 1e-3)))
            )
        else:
            centers = bursts
    else:
        bursts = (start, start + span / 2.0, end - 1e-3)
        # The formal Q1 warm starts place fixed centres ahead of burst time;
        # use that same causal offset for the three-UAV event candidates.
        offset = min(12.1, max(0.0, span / 2.0))
        centers = tuple(min(end, value + offset) for value in bursts)
    plan = construct_q3_plan(problem, burst_times_s=bursts, center_times_s=centers)
    return plan, verify_q3_plan(problem, plan)
