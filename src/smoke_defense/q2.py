"""Strong Q2 contracts for one-UAV, one-to-three-smoke plans."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from smoke_defense.coverage import CertificationStatus

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
    exposed_duration_s: float = Field(ge=0.0)
    maximum_exposed_interval_s: float = Field(ge=0.0)
    minimum_margin_m: float
    flight_distance_m: float = Field(ge=0.0)


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
