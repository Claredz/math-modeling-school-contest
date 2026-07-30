"""End-to-end Q1 solver across the approved inertial guidance sweep."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import cos, radians, sin
from pathlib import Path
from typing import Literal

import numpy as np

from smoke_defense.candidates import (
    RANKING_TIME_DECIMALS,
    Q1Candidate,
    candidate_rank_key,
    evaluate_smoke_against_detection,
    generate_q1_candidates,
)
from smoke_defense.certificates.q1 import (
    EarliestSmokeCertificate,
    SingleSmokeDurationCertificate,
    certify_earliest_smoke_availability,
    certify_single_smoke_duration,
)
from smoke_defense.constants import ProblemConstants, load_problem_constants
from smoke_defense.coverage import CertificationStatus
from smoke_defense.detection import DetectionSet, build_detection_set
from smoke_defense.dynamics import (
    MissileGuidanceSpec,
    MissileTrajectory,
    ShipMotion,
    integrate_inertial_missile,
    integrate_instantaneous_reference,
)
from smoke_defense.events import ClosedInterval
from smoke_defense.paths import UAV_SPEED_MPS
from smoke_defense.scenario import Scenario, scenario_hash
from smoke_defense.scenario_matrix import (
    DIRECTIONS,
    DISTANCES_M,
    generate_instantaneous_ablation_matrix,
    generate_q1_q3_matrix,
)

FROZEN_ASSUMPTION_IDS = tuple(
    f"A-{index:03d}" for index in range(1, 23)
)


@dataclass(frozen=True)
class Q1ScenarioResult:
    scenario_id: str
    scenario_hash: str
    assumption_ids: tuple[str, ...]
    model_layer: Literal["formal", "ablation"]
    guidance_model: str
    heading_response_rate_per_s: float | None
    max_turn_rate_deg_s: float | None
    hit_time_s: float | None
    detection: DetectionSet
    duration_certificate: SingleSmokeDurationCertificate
    causal_certificate: EarliestSmokeCertificate
    strict_status: CertificationStatus
    best_candidate: Q1Candidate | None


@dataclass(frozen=True)
class CandidateCrossValidation:
    scenario_id: str
    covered_duration_s: float
    exposed_duration_s: float
    maximum_exposed_interval_s: float
    strict_status: CertificationStatus


@dataclass(frozen=True)
class Q1GuidanceSweepResult:
    direction: str
    distance_m: float
    formal_results: tuple[Q1ScenarioResult, ...]
    ablation_result: Q1ScenarioResult
    cross_validation: tuple[CandidateCrossValidation, ...]
    parameter_sensitive: bool
    worst_case_scenario_id: str

    @property
    def reference_result(self) -> Q1ScenarioResult:
        for result in self.formal_results:
            if (
                result.heading_response_rate_per_s == 1.0
                and result.max_turn_rate_deg_s == 10.0
            ):
                return result
        raise RuntimeError("approved median guidance parameters are absent")


@dataclass(frozen=True)
class _InstantaneousHeadingAdapter:
    trajectory: MissileTrajectory
    ship: ShipMotion

    @property
    def start_time_s(self) -> float:
        return self.trajectory.start_time_s

    @property
    def end_time_s(self) -> float:
        return self.trajectory.end_time_s

    @property
    def hit_time_s(self) -> float | None:
        return self.trajectory.hit_time_s

    def position(self, time_s: float) -> np.ndarray:
        return self.trajectory.position(time_s)

    def heading(self, time_s: float) -> float:
        relative = self.ship.position(time_s) - self.position(time_s)
        return float(np.arctan2(relative[1], relative[0]))


def _body_to_world(
    body_position_m: tuple[float, float],
    ship_position_m: np.ndarray,
    ship_heading_rad: float,
) -> np.ndarray:
    rotation = np.array(
        [
            [cos(ship_heading_rad), -sin(ship_heading_rad)],
            [sin(ship_heading_rad), cos(ship_heading_rad)],
        ]
    )
    return ship_position_m + rotation @ np.asarray(body_position_m, dtype=float)


def _scenario_objects(
    scenario: Scenario,
    constants: ProblemConstants,
) -> tuple[ShipMotion, np.ndarray, float]:
    ship = ShipMotion(
        scenario.ship.initial_position_world_m,
        radians(scenario.ship.heading_deg),
        constants.ship.speed_mps,
    )
    missile = scenario.missiles[0]
    appearance_time_s = missile.appearance_time_s
    if missile.initial_position_world_m is not None:
        missile_position = np.asarray(
            missile.initial_position_world_m,
            dtype=float,
        )
    else:
        if missile.initial_position_at_appearance_body_m is None:
            raise RuntimeError("validated scenario lost its missile position")
        missile_position = _body_to_world(
            missile.initial_position_at_appearance_body_m,
            ship.position(appearance_time_s),
            ship.heading_rad,
        )
    return ship, missile_position, appearance_time_s


def _integrate_scenario(
    scenario: Scenario,
    constants: ProblemConstants,
) -> tuple[ShipMotion, MissileTrajectory | _InstantaneousHeadingAdapter]:
    ship, initial_position, appearance_time_s = _scenario_objects(
        scenario,
        constants,
    )
    missile = scenario.missiles[0]
    missile_speed = (
        missile.speed_override_mps
        if missile.speed_override_mps is not None
        else constants.missile.nominal_speed_mps
    )
    initial_distance = float(
        np.linalg.norm(initial_position - ship.position(appearance_time_s))
    )
    integration_horizon_s = max(
        120.0,
        2.0
        * initial_distance
        / max(1.0, missile_speed - constants.ship.speed_mps),
    )
    final_time_s = appearance_time_s + integration_horizon_s
    if missile.guidance_model == "inertial_pure_pursuit":
        if (
            missile.heading_response_rate_per_s is None
            or missile.max_turn_rate_deg_s is None
        ):
            raise RuntimeError("formal missile lacks inertial parameters")
        trajectory = integrate_inertial_missile(
            initial_position,
            appearance_time_s,
            ship.position,
            MissileGuidanceSpec(
                missile_speed,
                missile.heading_response_rate_per_s,
                missile.max_turn_rate_deg_s,
            ),
            t_final_s=final_time_s,
            hit_radius_m=constants.ship.effective_radius_m,
        )
        return ship, trajectory
    reference = integrate_instantaneous_reference(
        initial_position,
        appearance_time_s,
        ship.position,
        speed_mps=missile_speed,
        t_final_s=final_time_s,
        hit_radius_m=constants.ship.effective_radius_m,
    )
    return ship, _InstantaneousHeadingAdapter(reference, ship)


def solve_q1_scenario(
    scenario: Scenario,
    constants: ProblemConstants | None = None,
) -> Q1ScenarioResult:
    constants = constants or load_problem_constants()
    if not np.isclose(
        constants.uav.speed_mps,
        UAV_SPEED_MPS,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "the frozen Q1 model contract requires UAV speed 28 m/s"
        )
    ship, trajectory = _integrate_scenario(scenario, constants)
    missile = scenario.missiles[0]
    missile_speed_mps = (
        missile.speed_override_mps
        if missile.speed_override_mps is not None
        else constants.missile.nominal_speed_mps
    )
    distance_margin_lipschitz_mps = missile_speed_mps + ship.speed_mps
    fov_margin_lipschitz_rad_s = 0.0
    if missile.guidance_model == "inertial_pure_pursuit":
        if missile.max_turn_rate_deg_s is None:
            raise RuntimeError("formal missile lacks a turn-rate bound")
        fov_margin_lipschitz_rad_s = (
            distance_margin_lipschitz_mps
            / constants.ship.effective_radius_m
            + radians(missile.max_turn_rate_deg_s)
        )
    detection = build_detection_set(
        trajectory,
        ship.position,
        detection_range_m=constants.missile.detection_range_m,
        field_of_view_half_angle_rad=radians(
            constants.missile.field_of_view_half_angle_deg
        ),
        distance_margin_lipschitz_mps=distance_margin_lipschitz_mps,
        fov_margin_lipschitz_rad_s=fov_margin_lipschitz_rad_s,
        event_scan_step_s=0.02,
    )
    duration_certificate = certify_single_smoke_duration(
        detection.components,
        ship_speed_mps=constants.ship.speed_mps,
        maximum_smoke_radius_m=constants.smoke.maximum_radius_m,
        ship_radius_m=constants.ship.effective_radius_m,
    )
    causal_certificate = certify_earliest_smoke_availability(
        detection.components,
        command_time_s=0.0,
        minimum_release_response_s=(
            constants.countermeasure.release_response_min_s
        ),
        detonation_delay_s=constants.countermeasure.detonation_delay_s,
    )
    candidates = generate_q1_candidates(
        ship=ship,
        detection=detection,
        uav_available_time_s=scenario.uavs[0].available_time_s,
        operation_radius_m=constants.uav.operation_radius_m,
        minimum_release_response_s=(
            constants.countermeasure.release_response_min_s
        ),
        detonation_delay_s=constants.countermeasure.detonation_delay_s,
        maximum_smoke_radius_m=constants.smoke.maximum_radius_m,
        smoke_hold_duration_s=constants.smoke.hold_duration_s,
        smoke_decay_duration_s=constants.smoke.decay_duration_s,
        ship_radius_m=constants.ship.effective_radius_m,
    )
    best_candidate = max(candidates, key=candidate_rank_key, default=None)
    analytic_statuses = (
        duration_certificate.status,
        causal_certificate.status,
    )
    if CertificationStatus.CERTIFIED_INFEASIBLE in analytic_statuses:
        strict_status = CertificationStatus.CERTIFIED_INFEASIBLE
    elif best_candidate is None:
        strict_status = CertificationStatus.INDETERMINATE
    else:
        strict_status = best_candidate.strict_status
    return Q1ScenarioResult(
        scenario_id=scenario.scenario_id,
        scenario_hash=scenario_hash(scenario, constants=constants),
        assumption_ids=FROZEN_ASSUMPTION_IDS,
        model_layer=scenario.model_layer,
        guidance_model=missile.guidance_model,
        heading_response_rate_per_s=missile.heading_response_rate_per_s,
        max_turn_rate_deg_s=missile.max_turn_rate_deg_s,
        hit_time_s=trajectory.hit_time_s,
        detection=detection,
        duration_certificate=duration_certificate,
        causal_certificate=causal_certificate,
        strict_status=strict_status,
        best_candidate=best_candidate,
    )


def _cross_validate_reference(
    formal_results: tuple[Q1ScenarioResult, ...],
    reference: Q1ScenarioResult,
    ship: ShipMotion,
    ship_radius_m: float,
) -> tuple[CandidateCrossValidation, ...]:
    if reference.best_candidate is None or reference.best_candidate.smoke is None:
        return ()
    validation = []
    for result in formal_results:
        (
            _covered,
            covered_duration,
            exposed_duration,
            maximum_exposed,
            _minimum_margin,
            strict_status,
        ) = evaluate_smoke_against_detection(
            ship=ship,
            smoke=reference.best_candidate.smoke,
            detection=result.detection,
            ship_radius_m=ship_radius_m,
        )
        validation.append(
            CandidateCrossValidation(
                scenario_id=result.scenario_id,
                covered_duration_s=covered_duration,
                exposed_duration_s=exposed_duration,
                maximum_exposed_interval_s=maximum_exposed,
                strict_status=strict_status,
            )
        )
    return tuple(validation)


def _worst_case_scenario_id(
    cross_validation: tuple[CandidateCrossValidation, ...],
    *,
    fallback_scenario_id: str,
) -> str:
    if not cross_validation:
        return fallback_scenario_id
    return min(
        cross_validation,
        key=lambda item: (
            round(
                item.covered_duration_s,
                RANKING_TIME_DECIMALS,
            ),
            -item.maximum_exposed_interval_s,
        ),
    ).scenario_id


def solve_q1_guidance_sweep(
    *,
    direction: str,
    distance_m: float,
) -> Q1GuidanceSweepResult:
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction: {direction}")
    if distance_m not in DISTANCES_M:
        raise ValueError(f"distance is not in the approved matrix: {distance_m}")
    prefix = f"q1_q3_{direction}_d{int(distance_m)}_"
    formal_scenarios = tuple(
        scenario
        for scenario in generate_q1_q3_matrix()
        if scenario.scenario_id.startswith(prefix)
    )
    ablation_id = f"ablation_{direction}_d{int(distance_m)}"
    ablation_scenario = next(
        scenario
        for scenario in generate_instantaneous_ablation_matrix()
        if scenario.scenario_id == ablation_id
    )
    constants = load_problem_constants()
    formal_results = tuple(
        solve_q1_scenario(scenario, constants) for scenario in formal_scenarios
    )
    ablation_result = solve_q1_scenario(ablation_scenario, constants)
    reference = next(
        result
        for result in formal_results
        if result.heading_response_rate_per_s == 1.0
        and result.max_turn_rate_deg_s == 10.0
    )
    ship, _initial, _appearance = _scenario_objects(
        formal_scenarios[0],
        constants,
    )
    cross_validation = _cross_validate_reference(
        formal_results,
        reference,
        ship,
        constants.ship.effective_radius_m,
    )
    covered_values = [
        item.covered_duration_s for item in cross_validation
    ]
    detection_values = [result.detection.duration_s for result in formal_results]
    component_counts = {
        len(result.detection.components) for result in formal_results
    }
    coverage_sensitive = bool(covered_values) and (
        max(covered_values) - min(covered_values) > 1e-4
    )
    parameter_sensitive = (
        coverage_sensitive
        or (max(detection_values) - min(detection_values) > 1e-4)
        or len(component_counts) > 1
    )
    worst_case_scenario_id = _worst_case_scenario_id(
        cross_validation,
        fallback_scenario_id=reference.scenario_id,
    )
    return Q1GuidanceSweepResult(
        direction=direction,
        distance_m=distance_m,
        formal_results=formal_results,
        ablation_result=ablation_result,
        cross_validation=cross_validation,
        parameter_sensitive=parameter_sensitive,
        worst_case_scenario_id=worst_case_scenario_id,
    )


def solve_all_q1_sweeps() -> tuple[Q1GuidanceSweepResult, ...]:
    return tuple(
        solve_q1_guidance_sweep(direction=direction, distance_m=distance_m)
        for direction in DIRECTIONS
        for distance_m in DISTANCES_M
    )


def _interval_dict(interval: ClosedInterval) -> dict[str, float]:
    return {"start_s": interval.start_s, "end_s": interval.end_s}


def _candidate_dict(candidate: Q1Candidate | None) -> dict | None:
    if candidate is None:
        return None
    path_nodes = []
    if candidate.path is not None:
        path_nodes.append(
            {
                "time_s": candidate.path.takeoff_time_s,
                "position_m": candidate.path.position(
                    candidate.path.takeoff_time_s
                ).tolist(),
            }
        )
        path_nodes.extend(
            {
                "time_s": segment.end_time_s,
                "position_m": segment.end_position_m.tolist(),
            }
            for segment in candidate.path.segments
        )
    return {
        "strict_status": candidate.strict_status.value,
        "command_time_s": candidate.command_time_s,
        "takeoff_time_s": candidate.takeoff_time_s,
        "release_time_s": candidate.release_time_s,
        "burst_time_s": candidate.burst_time_s,
        "coverage_center_time_s": candidate.coverage_center_time_s,
        "release_position_m": (
            candidate.release_position_m.tolist()
            if candidate.release_position_m is not None
            else None
        ),
        "burst_center_m": (
            candidate.burst_center_m.tolist()
            if candidate.burst_center_m is not None
            else None
        ),
        "covered_intervals": [
            _interval_dict(interval) for interval in candidate.covered_intervals
        ],
        "covered_duration_s": candidate.covered_duration_s,
        "exposed_duration_s": candidate.exposed_duration_s,
        "maximum_exposed_interval_s": candidate.maximum_exposed_interval_s,
        "minimum_margin_m": candidate.minimum_margin_m,
        "flight_distance_m": candidate.flight_distance_m,
        "reachability_status": candidate.reachability_status,
        "path_nodes": path_nodes,
    }


def _scenario_result_dict(result: Q1ScenarioResult) -> dict:
    return {
        "scenario_id": result.scenario_id,
        "scenario_hash": result.scenario_hash,
        "assumption_ids": list(result.assumption_ids),
        "model_layer": result.model_layer,
        "guidance_model": result.guidance_model,
        "heading_response_rate_per_s": result.heading_response_rate_per_s,
        "max_turn_rate_deg_s": result.max_turn_rate_deg_s,
        "hit_time_s": result.hit_time_s,
        "detection_components": [
            _interval_dict(component) for component in result.detection.components
        ],
        "detection_events": [
            {"time_s": event.time_s, "kind": event.kind.value}
            for event in result.detection.source_events
        ],
        "detection_duration_s": result.detection.duration_s,
        "duration_certificate": {
            "status": result.duration_certificate.status.value,
            "limit_s": result.duration_certificate.limit_s,
            "longest_component_s": (
                result.duration_certificate.longest_component_s
            ),
            "reason": result.duration_certificate.reason,
        },
        "causal_certificate": {
            "status": result.causal_certificate.status.value,
            "earliest_burst_time_s": (
                result.causal_certificate.earliest_burst_time_s
            ),
            "unavoidable_exposure_s": (
                result.causal_certificate.unavoidable_exposure_s
            ),
            "reason": result.causal_certificate.reason,
        },
        "strict_status": result.strict_status.value,
        "best_candidate": _candidate_dict(result.best_candidate),
    }


def _sweep_dict(sweep: Q1GuidanceSweepResult) -> dict:
    return {
        "direction": sweep.direction,
        "distance_m": sweep.distance_m,
        "formal_results": [
            _scenario_result_dict(result) for result in sweep.formal_results
        ],
        "ablation_result": _scenario_result_dict(sweep.ablation_result),
        "reference_scenario_id": sweep.reference_result.scenario_id,
        "cross_validation": [
            {
                "scenario_id": item.scenario_id,
                "covered_duration_s": item.covered_duration_s,
                "exposed_duration_s": item.exposed_duration_s,
                "maximum_exposed_interval_s": (
                    item.maximum_exposed_interval_s
                ),
                "strict_status": item.strict_status.value,
            }
            for item in sweep.cross_validation
        ],
        "parameter_sensitive": sweep.parameter_sensitive,
        "worst_case_scenario_id": sweep.worst_case_scenario_id,
    }


def write_q1_sweep_result(
    sweeps: tuple[Q1GuidanceSweepResult, ...],
    output_path: str | Path,
    *,
    git_sha: str,
    random_seed: int,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "random_seed": random_seed,
        "model_contract_version": "v0.2",
        "assumption_register_version": "v0.3",
        "constants_version": load_problem_constants().constants_version,
        "formal_guidance_model": "inertial_pure_pursuit",
        "ablation_guidance_model": "instantaneous_pure_pursuit",
        "sweeps": [_sweep_dict(sweep) for sweep in sweeps],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def write_q1_markdown_summary(
    sweeps: tuple[Q1GuidanceSweepResult, ...],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Q1 惯性纯追踪计算结果",
        "",
        "> 正式层采用一阶航向惯性纯追踪；瞬时纯追踪只作为消融对照。",
        "",
        "单个固定烟幕完整覆盖的解析上限为 "
        f"`{80.0 / 7.71:.6f} s`。表中方案已从运动舰艇实时位置起飞，"
        "并通过 28 m/s 航段、98 m 投弹惯性段和 12 km 作战半径检查。",
        "",
        "| 方位 | 初距/m | 中值参数探测时长/s | 最佳完整覆盖/s | 裸露/s | "
        "严格状态 | 参数敏感 | 最坏参数场景 |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for sweep in sweeps:
        reference = sweep.reference_result
        candidate = reference.best_candidate
        covered = candidate.covered_duration_s if candidate else 0.0
        exposed = candidate.exposed_duration_s if candidate else reference.detection.duration_s
        lines.append(
            f"| {sweep.direction} | {sweep.distance_m:.0f} | "
            f"{reference.detection.duration_s:.6f} | {covered:.6f} | "
            f"{exposed:.6f} | {reference.strict_status.value} | "
            f"{'是' if sweep.parameter_sensitive else '否'} | "
            f"`{sweep.worst_case_scenario_id}` |"
        )
    strict_infeasible = sum(
        sweep.reference_result.strict_status
        is CertificationStatus.CERTIFIED_INFEASIBLE
        for sweep in sweeps
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 共汇总 {len(sweeps)} 组基础几何；其中中值参数下 "
            f"{strict_infeasible} 组严格目标被认证为不可行。",
            "- 不可行并不等于无效：结果仍按词典序给出最大完整覆盖时长、"
            "最大连续裸露、最小裕度与飞行距离。",
            "- `parameter_sensitive` 由 9 组惯性参数的探测/覆盖结果范围判定；"
            "它不把中值参数误写成题面事实。",
            "- 逐场景哈希、命中/探测事件、解析证书、候选路径与消融结果见 "
            "[q1_sweep_results.json](q1_sweep_results.json)。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
