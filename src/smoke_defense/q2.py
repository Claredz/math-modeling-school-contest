"""Strong Q2 contracts for one-UAV, one-to-three-smoke plans."""

from __future__ import annotations

from math import cos, sin
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from smoke_defense.candidates import (
    build_shipborne_release_path,
    evaluate_smoke_against_detection,
    generate_q1_candidates,
)
from smoke_defense.coverage import CertificationStatus
from smoke_defense.detection import DetectionSet
from smoke_defense.dynamics import ShipMotion
from smoke_defense.path_constraints import certify_operation_radius
from smoke_defense.smoke import SmokeCloud

Vector2 = tuple[float, float]
ModelLayer = Literal["formal", "ablation"]


class FrozenQ2Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathNode(FrozenQ2Model):
    time_s: float = Field(ge=0.0)
    position_m: Vector2


class SmokeReleaseEvent(FrozenQ2Model):
    candidate_id: str = Field(min_length=1)
    command_time_s: float = Field(ge=0.0)
    release_time_s: float = Field(ge=0.0)
    release_position_m: Vector2
    release_heading_unit: Vector2
    burst_time_s: float = Field(ge=0.0)
    burst_center_m: Vector2

    @model_validator(mode="after")
    def validate_release_geometry(self) -> SmokeReleaseEvent:
        if self.release_time_s < self.command_time_s + 2.0 - 1e-9:
            raise ValueError("release must respect the 2 s response time")
        if not np.isclose(
            self.burst_time_s,
            self.release_time_s + 3.5,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("burst must occur 3.5 s after release")
        heading = np.asarray(self.release_heading_unit, dtype=float)
        if not np.isclose(
            np.linalg.norm(heading),
            1.0,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("release heading must be a unit vector")
        expected_burst = (
            np.asarray(self.release_position_m, dtype=float)
            + 98.0 * heading
        )
        if not np.allclose(
            expected_burst,
            self.burst_center_m,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "burst centre must follow the 3.5 s inertial trajectory"
            )
        return self


class OrderedReleasePlan(FrozenQ2Model):
    takeoff_time_s: float = Field(ge=0.0)
    takeoff_position_m: Vector2
    path_nodes: tuple[PathNode, ...] = Field(min_length=2)
    releases: tuple[SmokeReleaseEvent, ...] = Field(
        min_length=1,
        max_length=3,
    )
    adjacent_release_intervals_s: tuple[float, ...]
    flight_distance_m: float = Field(ge=0.0)
    continue_until_s: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_ordered_releases(self) -> OrderedReleasePlan:
        expected_intervals = tuple(
            right.release_time_s - left.release_time_s
            for left, right in zip(
                self.releases[:-1],
                self.releases[1:],
                strict=True,
            )
        )
        if any(interval_s < 1.0 - 1e-9 for interval_s in expected_intervals):
            raise ValueError("adjacent releases must be at least 1 s apart")
        if len(self.adjacent_release_intervals_s) != len(expected_intervals):
            raise ValueError("adjacent release intervals have the wrong length")
        if not np.allclose(
            self.adjacent_release_intervals_s,
            expected_intervals,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("adjacent release intervals do not match events")
        if not np.isclose(
            self.path_nodes[0].time_s,
            self.takeoff_time_s,
            rtol=0.0,
            atol=1e-9,
        ) or not np.allclose(
            self.path_nodes[0].position_m,
            self.takeoff_position_m,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("first path node must be the actual takeoff state")
        if self.continue_until_s <= self.releases[-1].release_time_s:
            raise ValueError("path must continue after the final release")
        if self.path_nodes[-1].time_s < self.continue_until_s - 1e-9:
            raise ValueError("path nodes do not reach continue_until_s")
        for release in self.releases:
            if not any(
                np.isclose(
                    node.time_s,
                    release.release_time_s,
                    rtol=0.0,
                    atol=1e-9,
                )
                and np.allclose(
                    node.position_m,
                    release.release_position_m,
                    rtol=0.0,
                    atol=1e-8,
                )
                for node in self.path_nodes
            ):
                raise ValueError("every release event must lie on a path node")
        return self


class MultiSmokeCandidate(FrozenQ2Model):
    candidate_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_hash: str = Field(min_length=1)
    constants_hash: str = Field(min_length=1)
    assumption_ids: tuple[str, ...] = Field(min_length=1)
    guidance_model: str = Field(min_length=1)
    model_layer: ModelLayer
    heading_response_rate_per_s: float | None = Field(default=None, gt=0.0)
    max_turn_rate_deg_s: float | None = Field(default=None, gt=0.0)
    takeoff_time_s: float = Field(ge=0.0)
    coverage_center_time_s: float = Field(ge=0.0)
    longitudinal_offset_m: float
    lateral_offset_m: float
    release: SmokeReleaseEvent
    maximum_radius_m: float = Field(gt=0.0)
    hold_duration_s: float = Field(ge=0.0)
    decay_duration_s: float = Field(gt=0.0)
    path_start_position_m: Vector2
    path_end_position_m: Vector2
    reachability_status: Literal[
        "certified_feasible",
        "certified_infeasible",
        "indeterminate_at_tolerance",
    ]
    single_coverage_status: CertificationStatus
    covered_duration_s: float = Field(ge=0.0)
    covered_intervals_s: tuple[tuple[float, float], ...]
    exposed_duration_s: float = Field(ge=0.0)
    maximum_exposed_interval_s: float = Field(ge=0.0)
    minimum_margin_m: float
    flight_distance_m: float = Field(ge=0.0)


class Q2CandidateLibrary(FrozenQ2Model):
    generated_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    candidates: tuple[MultiSmokeCandidate, ...]


class CandidatePruningResult(FrozenQ2Model):
    input_count: int = Field(ge=0)
    retained_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    retained_candidates: tuple[MultiSmokeCandidate, ...]


class CoverageModeRecord(FrozenQ2Model):
    start_time_s: float
    end_time_s: float
    active_smoke_indices: tuple[int, ...]
    status: CertificationStatus
    maximum_gap_lower_bound_m: float
    maximum_gap_upper_bound_m: float


class MultiSmokeCertificate(FrozenQ2Model):
    status: CertificationStatus
    modes: tuple[CoverageModeRecord, ...]
    maximum_gap_lower_bound_m: float
    maximum_gap_upper_bound_m: float
    minimum_margin_m: float
    total_exposed_duration_s: float = Field(ge=0.0)
    maximum_exposed_interval_start_s: float | None = None
    maximum_exposed_interval_end_s: float | None = None
    maximum_exposed_interval_s: float = Field(ge=0.0)
    witness_time_s: float | None = None
    witness_m: Vector2 | None = None
    spatial_tolerance_m: float = Field(gt=0.0)
    time_tolerance_s: float = Field(gt=0.0)


class OperationRadiusCertificate(FrozenQ2Model):
    status: Literal["certified_feasible", "certified_infeasible"]
    maximum_distance_m: float = Field(ge=0.0)
    limit_m: float = Field(gt=0.0)
    critical_time_s: float = Field(ge=0.0)


class IndependentCoverageBaseline(FrozenQ2Model):
    covered_duration_s: float = Field(ge=0.0)
    total_exposed_duration_s: float = Field(ge=0.0)
    maximum_exposed_interval_s: float = Field(ge=0.0)
    union_gain_s: float


class SolverTrace(FrozenQ2Model):
    method: str = Field(min_length=1)
    candidate_count: int = Field(ge=0)
    retained_combination_count: int = Field(ge=0)
    pruned_combination_count: int = Field(ge=0)
    random_seed: int


class VerifierTrace(FrozenQ2Model):
    method: str = Field(min_length=1)
    initial_polygon_sides: int = Field(ge=4)
    maximum_polygon_sides: int = Field(ge=4)
    spatial_tolerance_m: float = Field(gt=0.0)
    time_tolerance_s: float = Field(gt=0.0)


class MultiSmokePlan(FrozenQ2Model):
    scenario_id: str = Field(min_length=1)
    scenario_hash: str = Field(min_length=1)
    constants_hash: str = Field(min_length=1)
    assumption_ids: tuple[str, ...] = Field(min_length=1)
    guidance_model: str = Field(min_length=1)
    model_layer: ModelLayer
    heading_response_rate_per_s: float | None = Field(default=None, gt=0.0)
    max_turn_rate_deg_s: float | None = Field(default=None, gt=0.0)
    actual_takeoff_time_s: float = Field(ge=0.0)
    ordered_path: OrderedReleasePlan
    selected_smokes: tuple[MultiSmokeCandidate, ...] = Field(
        min_length=1,
        max_length=3,
    )
    adjacent_release_intervals_s: tuple[float, ...]
    coverage_certificate: MultiSmokeCertificate
    maximum_joint_gap_m: float
    minimum_joint_margin_m: float
    total_exposed_duration_s: float = Field(ge=0.0)
    maximum_exposed_interval_start_s: float | None = None
    maximum_exposed_interval_end_s: float | None = None
    maximum_exposed_interval_s: float = Field(ge=0.0)
    smoke_count: int = Field(ge=1, le=3)
    uav_total_distance_m: float = Field(ge=0.0)
    operation_radius_certificate: OperationRadiusCertificate
    independent_coverage_baseline: IndependentCoverageBaseline
    solver_trace: SolverTrace
    verifier_trace: VerifierTrace

    @model_validator(mode="after")
    def validate_plan_consistency(self) -> MultiSmokePlan:
        if self.smoke_count != len(self.selected_smokes):
            raise ValueError("smoke_count must match selected smokes")
        if self.adjacent_release_intervals_s != (
            self.ordered_path.adjacent_release_intervals_s
        ):
            raise ValueError("plan release intervals must match the path")
        if not np.isclose(
            self.actual_takeoff_time_s,
            self.ordered_path.takeoff_time_s,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("plan takeoff time must match the path")
        return self


class Q2ScenarioResult(FrozenQ2Model):
    scenario_id: str
    scenario_hash: str
    model_layer: ModelLayer
    guidance_model: str
    heading_response_rate_per_s: float | None
    max_turn_rate_deg_s: float | None
    candidate_library_size: int = Field(ge=0)
    pruned_candidate_count: int = Field(ge=0)
    selected_plan: MultiSmokePlan | None
    strict_status: CertificationStatus


class Q2SweepResult(FrozenQ2Model):
    direction: str
    distance_m: float = Field(gt=0.0)
    formal_results: tuple[Q2ScenarioResult, ...]
    ablation_result: Q2ScenarioResult
    worst_case_scenario_id: str
    parameter_sensitive: bool


def _ship_axes(ship: ShipMotion) -> tuple[np.ndarray, np.ndarray]:
    longitudinal = np.array(
        [cos(ship.heading_rad), sin(ship.heading_rad)],
        dtype=float,
    )
    lateral = np.array([-longitudinal[1], longitudinal[0]])
    return longitudinal, lateral


def generate_q2_candidate_library(
    *,
    ship: ShipMotion,
    detection: DetectionSet,
    uav_available_time_s: float,
    scenario_id: str,
    scenario_hash: str,
    constants_hash: str,
    assumption_ids: tuple[str, ...],
    guidance_model: str,
    model_layer: ModelLayer,
    heading_response_rate_per_s: float | None,
    max_turn_rate_deg_s: float | None,
    operation_radius_m: float = 12000.0,
    command_time_s: float = 0.0,
    minimum_release_response_s: float = 2.0,
    detonation_delay_s: float = 3.5,
    maximum_smoke_radius_m: float = 120.0,
    smoke_hold_duration_s: float = 18.0,
    smoke_decay_duration_s: float = 5.0,
    ship_radius_m: float = 80.0,
    longitudinal_offsets_m: tuple[float, ...] = (0.0,),
    lateral_offsets_m: tuple[float, ...] = (-50.0, 0.0, 50.0),
) -> Q2CandidateLibrary:
    """Expand all Q1 structural candidates without selecting only the winner."""

    base_candidates = generate_q1_candidates(
        ship=ship,
        detection=detection,
        uav_available_time_s=uav_available_time_s,
        operation_radius_m=operation_radius_m,
        command_time_s=command_time_s,
        minimum_release_response_s=minimum_release_response_s,
        detonation_delay_s=detonation_delay_s,
        maximum_smoke_radius_m=maximum_smoke_radius_m,
        smoke_hold_duration_s=smoke_hold_duration_s,
        smoke_decay_duration_s=smoke_decay_duration_s,
        ship_radius_m=ship_radius_m,
    )
    longitudinal_axis, lateral_axis = _ship_axes(ship)
    candidates: list[MultiSmokeCandidate] = []
    generated_count = 0
    rejected_count = 0
    for base_index, base in enumerate(base_candidates):
        if base.burst_center_m is None:
            continue
        for longitudinal_offset_m in longitudinal_offsets_m:
            for lateral_offset_m in lateral_offsets_m:
                generated_count += 1
                burst_center = (
                    base.burst_center_m
                    + longitudinal_offset_m * longitudinal_axis
                    + lateral_offset_m * lateral_axis
                )
                try:
                    path, release_position, release_heading = (
                        build_shipborne_release_path(
                            ship=ship,
                            takeoff_time_s=base.takeoff_time_s,
                            release_time_s=base.release_time_s,
                            burst_center_m=burst_center,
                            detonation_delay_s=detonation_delay_s,
                        )
                    )
                    radius_certificate = certify_operation_radius(
                        path,
                        operation_radius_m=operation_radius_m,
                    )
                    if radius_certificate.status != "certified_feasible":
                        rejected_count += 1
                        continue
                    smoke = SmokeCloud(
                        burst_time_s=base.burst_time_s,
                        burst_center_m=burst_center,
                        maximum_radius_m=maximum_smoke_radius_m,
                        hold_duration_s=smoke_hold_duration_s,
                        decay_duration_s=smoke_decay_duration_s,
                    )
                    (
                        covered_intervals,
                        covered_duration_s,
                        exposed_duration_s,
                        maximum_exposed_interval_s,
                        minimum_margin_m,
                        strict_status,
                    ) = evaluate_smoke_against_detection(
                        ship=ship,
                        smoke=smoke,
                        detection=detection,
                        ship_radius_m=ship_radius_m,
                    )
                    release = SmokeReleaseEvent(
                        candidate_id=(
                            f"{scenario_id}-q2-{base_index}-"
                            f"{longitudinal_offset_m:g}-"
                            f"{lateral_offset_m:g}"
                        ),
                        command_time_s=base.command_time_s,
                        release_time_s=base.release_time_s,
                        release_position_m=tuple(release_position),
                        release_heading_unit=tuple(release_heading),
                        burst_time_s=base.burst_time_s,
                        burst_center_m=tuple(burst_center),
                    )
                    candidates.append(
                        MultiSmokeCandidate(
                            candidate_id=release.candidate_id,
                            scenario_id=scenario_id,
                            scenario_hash=scenario_hash,
                            constants_hash=constants_hash,
                            assumption_ids=assumption_ids,
                            guidance_model=guidance_model,
                            model_layer=model_layer,
                            heading_response_rate_per_s=(
                                heading_response_rate_per_s
                            ),
                            max_turn_rate_deg_s=max_turn_rate_deg_s,
                            takeoff_time_s=base.takeoff_time_s,
                            coverage_center_time_s=(
                                base.coverage_center_time_s
                            ),
                            longitudinal_offset_m=longitudinal_offset_m,
                            lateral_offset_m=lateral_offset_m,
                            release=release,
                            maximum_radius_m=maximum_smoke_radius_m,
                            hold_duration_s=smoke_hold_duration_s,
                            decay_duration_s=smoke_decay_duration_s,
                            path_start_position_m=tuple(
                                ship.position(base.takeoff_time_s)
                            ),
                            path_end_position_m=tuple(release_position),
                            reachability_status=radius_certificate.status,
                            single_coverage_status=strict_status,
                            covered_duration_s=covered_duration_s,
                            covered_intervals_s=tuple(
                                (interval.start_s, interval.end_s)
                                for interval in covered_intervals
                            ),
                            exposed_duration_s=exposed_duration_s,
                            maximum_exposed_interval_s=(
                                maximum_exposed_interval_s
                            ),
                            minimum_margin_m=minimum_margin_m,
                            flight_distance_m=path.flight_distance_m,
                        )
                    )
                except ValueError:
                    rejected_count += 1
    pruning = prune_q2_candidates(tuple(candidates))
    return Q2CandidateLibrary(
        generated_count=generated_count,
        rejected_count=(
            rejected_count + pruning.duplicate_count
        ),
        candidates=pruning.retained_candidates,
    )


def _physical_candidate_key(candidate: MultiSmokeCandidate) -> tuple:
    release = candidate.release

    def rounded(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(round(value, 9) for value in values)

    return (
        round(candidate.takeoff_time_s, 9),
        round(release.release_time_s, 9),
        rounded(release.release_position_m),
        rounded(release.release_heading_unit),
        round(release.burst_time_s, 9),
        rounded(release.burst_center_m),
        round(candidate.maximum_radius_m, 9),
        round(candidate.hold_duration_s, 9),
        round(candidate.decay_duration_s, 9),
    )


def prune_q2_candidates(
    candidates: tuple[MultiSmokeCandidate, ...],
) -> CandidatePruningResult:
    """Remove only physically duplicate events; retain incomplete smokes."""

    retained_by_key: dict[tuple, MultiSmokeCandidate] = {}
    for candidate in candidates:
        key = _physical_candidate_key(candidate)
        incumbent = retained_by_key.get(key)
        if (
            incumbent is None
            or candidate.flight_distance_m < incumbent.flight_distance_m
        ):
            retained_by_key[key] = candidate
    retained = tuple(
        sorted(
            retained_by_key.values(),
            key=lambda candidate: (
                candidate.release.release_time_s,
                candidate.release.burst_center_m,
                candidate.candidate_id,
            ),
        )
    )
    return CandidatePruningResult(
        input_count=len(candidates),
        retained_count=len(retained),
        duplicate_count=len(candidates) - len(retained),
        retained_candidates=retained,
    )
