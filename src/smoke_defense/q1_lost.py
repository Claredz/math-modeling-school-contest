"""Q1 optimizer for the smoke-loss and turn-inertia improved model."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, radians

import numpy as np

from smoke_defense.candidates import build_shipborne_release_path
from smoke_defense.constants import ProblemConstants, load_problem_constants
from smoke_defense.dynamics import ShipMotion
from smoke_defense.lost_guidance import (
    CoupledTrajectory,
    LostGuidanceSpec,
    simulate_lost_coupled_missile,
)
from smoke_defense.path_constraints import certify_operation_radius
from smoke_defense.paths import UAV_SPEED_MPS, ShipborneUavPath
from smoke_defense.scenario import Scenario
from smoke_defense.smoke import SmokeCloud


@dataclass(frozen=True)
class LostQ1Candidate:
    burst_time_s: float
    release_time_s: float
    takeoff_time_s: float
    release_position_m: np.ndarray
    burst_center_m: np.ndarray
    flight_distance_m: float
    trajectory: CoupledTrajectory
    path: ShipborneUavPath

    @property
    def successful_defense(self) -> bool:
        return self.trajectory.successful_defense


@dataclass(frozen=True)
class LostQ1OptimizationResult:
    scenario_id: str
    initial_heading_error_deg: float
    candidates: tuple[LostQ1Candidate, ...]
    best_candidate: LostQ1Candidate | None
    unique_optimum_on_search_grid: bool

    @property
    def feasible(self) -> bool:
        return bool(self.best_candidate and self.best_candidate.successful_defense)


def make_custom_q1_scenario(
    *,
    scenario_id: str,
    missile_position_world_m: tuple[float, float],
    heading_response_rate_per_s: float = 1.0,
    max_turn_rate_deg_s: float = 5.0,
    appearance_time_s: float = 0.0,
) -> Scenario:
    """Build a validated evaluator scenario from judge-editable parameters."""

    return Scenario.model_validate(
        {
            "schema_version": "1.0",
            "constants_version": "b-problem-v2",
            "scenario_id": scenario_id,
            "time_origin": "decision_start",
            "model_layer": "formal",
            "assumption_ids": [
                "A-001",
                "A-002",
                "A-003",
                "A-019",
                "A-020",
                "A-021",
                "A-023",
                "A-024",
                "A-025",
                "A-026",
            ],
            "ship": {
                "initial_position_world_m": (0.0, 0.0),
                "heading_deg": 0.0,
            },
            "uavs": [{"id": "U1", "available_time_s": 0.0}],
            "missiles": [
                {
                    "id": "M1",
                    "appearance_time_s": appearance_time_s,
                    "initial_position_world_m": missile_position_world_m,
                    "guidance_model": "inertial_pure_pursuit",
                    "heading_response_rate_per_s": heading_response_rate_per_s,
                    "max_turn_rate_deg_s": max_turn_rate_deg_s,
                    "optical_axis_model": "velocity_aligned",
                }
            ],
            "constraints": {"safe_distance_m": 100.0},
        }
    )


def _scenario_geometry(
    scenario: Scenario,
    constants: ProblemConstants,
) -> tuple[ShipMotion, np.ndarray, float]:
    ship = ShipMotion(
        scenario.ship.initial_position_world_m,
        radians(scenario.ship.heading_deg),
        constants.ship.speed_mps,
    )
    missile = scenario.missiles[0]
    appearance = missile.appearance_time_s
    if missile.initial_position_world_m is not None:
        position = np.asarray(missile.initial_position_world_m, dtype=float)
    else:
        body = np.asarray(missile.initial_position_at_appearance_body_m, dtype=float)
        heading = ship.heading_rad
        rotation = np.array(
            [
                [np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)],
            ]
        )
        position = ship.position(appearance) + rotation @ body
    return ship, position, appearance


def lost_candidate_rank(candidate: LostQ1Candidate) -> tuple[float, ...]:
    """Feasibility constraint followed by the single miss-distance objective."""

    return (
        float(candidate.successful_defense),
        round(candidate.trajectory.minimum_separation_m, 6),
    )


def solve_q1_lost_coupled(
    scenario: Scenario,
    *,
    initial_heading_error_deg: float,
    lost_turn_decay_time_s: float,
    reacquisition_confirm_s: float = 1.0,
    tracked_turn_time_constant_s: float = 0.5,
    burst_times_s: tuple[float, ...],
    final_time_s: float = 150.0,
    time_step_s: float = 0.05,
    constants: ProblemConstants | None = None,
) -> LostQ1OptimizationResult:
    """Search physically reachable burst times and certify permanent loss.

    ``unique_optimum_on_search_grid`` is deliberately scoped to the declared
    grid.  It is not presented as a proof of uniqueness over continuous time.
    """

    constants = constants or load_problem_constants()
    ship, missile_position, appearance = _scenario_geometry(scenario, constants)
    missile = scenario.missiles[0]
    if missile.heading_response_rate_per_s is None or missile.max_turn_rate_deg_s is None:
        raise ValueError("lost-coupled Q1 requires formal inertial guidance parameters")
    spec = LostGuidanceSpec(
        speed_mps=missile.speed_override_mps or constants.missile.nominal_speed_mps,
        heading_response_rate_per_s=missile.heading_response_rate_per_s,
        max_turn_rate_deg_s=missile.max_turn_rate_deg_s,
        tracked_turn_time_constant_s=tracked_turn_time_constant_s,
        lost_turn_decay_time_s=lost_turn_decay_time_s,
        reacquisition_confirm_s=reacquisition_confirm_s,
        detection_range_m=constants.missile.detection_range_m,
        field_of_view_half_angle_deg=constants.missile.field_of_view_half_angle_deg,
        hit_radius_m=constants.ship.effective_radius_m,
    )
    relative = ship.position(appearance) - missile_position
    initial_heading = atan2(relative[1], relative[0]) + radians(
        initial_heading_error_deg
    )
    takeoff = scenario.uavs[0].available_time_s
    candidates: list[LostQ1Candidate] = []
    for burst_time in sorted(set(float(value) for value in burst_times_s)):
        release_time = burst_time - constants.countermeasure.detonation_delay_s
        if release_time < constants.countermeasure.release_response_min_s:
            continue
        burst_center = ship.position(burst_time)
        try:
            path, release_position, _heading = build_shipborne_release_path(
                ship=ship,
                takeoff_time_s=takeoff,
                release_time_s=release_time,
                burst_center_m=burst_center,
                detonation_delay_s=constants.countermeasure.detonation_delay_s,
            )
        except ValueError:
            continue
        radius_certificate = certify_operation_radius(
            path, operation_radius_m=constants.uav.operation_radius_m
        )
        if radius_certificate.status != "certified_feasible":
            continue
        smoke = SmokeCloud(
            burst_time_s=burst_time,
            burst_center_m=burst_center,
            maximum_radius_m=constants.smoke.maximum_radius_m,
            hold_duration_s=constants.smoke.hold_duration_s,
            decay_duration_s=constants.smoke.decay_duration_s,
        )
        trajectory = simulate_lost_coupled_missile(
            initial_position_m=missile_position,
            appearance_time_s=appearance,
            ship=ship,
            smoke=smoke,
            spec=spec,
            final_time_s=final_time_s,
            time_step_s=time_step_s,
            initial_heading_rad=initial_heading,
        )
        candidates.append(
            LostQ1Candidate(
                burst_time_s=burst_time,
                release_time_s=release_time,
                takeoff_time_s=takeoff,
                release_position_m=release_position,
                burst_center_m=burst_center,
                flight_distance_m=UAV_SPEED_MPS * (release_time - takeoff),
                trajectory=trajectory,
                path=path,
            )
        )
    best = max(candidates, key=lost_candidate_rank, default=None)
    if best is None:
        unique = False
    else:
        best_rank = lost_candidate_rank(best)
        unique = sum(lost_candidate_rank(item) == best_rank for item in candidates) == 1
    return LostQ1OptimizationResult(
        scenario_id=scenario.scenario_id,
        initial_heading_error_deg=initial_heading_error_deg,
        candidates=tuple(candidates),
        best_candidate=best,
        unique_optimum_on_search_grid=unique,
    )
