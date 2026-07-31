"""Stage 5 Q1 rebuild with an explicit solver/verifier boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

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
