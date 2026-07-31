"""Run the formal Q1 rebuild matrix and write paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from smoke_defense.coverage import single_smoke_gap
from smoke_defense.lost_counterfactual import (
    LostCounterfactualParameters,
    simulate_lost_counterfactual,
)
from smoke_defense.q1 import _integrate_scenario
from smoke_defense.q1_rebuild import (
    Q1_METHODS,
    Q1MethodResult,
    benchmark_q1_methods,
    build_q1_problem,
    q1_verification_rank,
)
from smoke_defense.scenario_matrix import generate_q1_rebuild_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "q1_rebuild"
FIGURES = ROOT / "figures" / "q1_rebuild"


def _git_sha() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _method_payload(result: Q1MethodResult, runtime_s: float) -> dict:
    verification = result.verification
    return {
        "method": result.method,
        "seed": result.seed,
        "evaluation_budget": result.evaluation_budget,
        "evaluations": result.evaluations,
        "runtime_s": runtime_s,
        "native_success": result.native_success,
        "native_status": result.native_status,
        "verification_status": verification.status if verification else "unresolved",
        "covered_duration_s": verification.covered_duration_s if verification else 0.0,
        "maximum_exposure_s": verification.maximum_exposure_s if verification else 0.0,
    }


def _scenario_payload(scenario, result, method: str) -> dict:
    candidate = result.best_candidate
    verification = result.verification
    if candidate is None or verification is None:
        return {
            "scenario_id": scenario.scenario_id,
            "method": method,
            "verification_status": "unresolved",
            "covered_duration_s": 0.0,
            "exposed_duration_s": 0.0,
            "maximum_exposure_s": 0.0,
            "minimum_margin_m": None,
            "flight_distance_m": None,
        }
    return {
        "scenario_id": scenario.scenario_id,
        "method": method,
        "native_success": result.native_success,
        "native_status": result.native_status,
        "verification_status": verification.status,
        "verification_reason": verification.reason,
        "covered_duration_s": verification.covered_duration_s,
        "exposed_duration_s": verification.exposed_duration_s,
        "maximum_exposure_s": verification.maximum_exposure_s,
        "minimum_margin_m": verification.minimum_margin_m,
        "flight_distance_m": verification.flight_distance_m,
        "command_time_s": candidate.command_time_s,
        "drop_time_s": candidate.drop_time_s,
        "burst_time_s": candidate.burst_time_s,
        "center_time_s": candidate.center_time_s,
        "drop_position_m": candidate.drop_position_m.tolist(),
        "burst_center_m": candidate.smoke.burst_center_m.tolist(),
        "detection_components": [
            {"start_s": item.start_s, "end_s": item.end_s}
            for item in build_q1_problem(scenario).detection.components
        ],
    }


def _write_figures(problem, candidate, payloads: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if candidate is None:
        return
    detection = problem.detection.components
    fig, ax = plt.subplots(figsize=(10, 4))
    for interval in detection:
        ax.axvspan(interval.start_s, interval.end_s, color="#dbeafe", alpha=0.8)
    for interval in payloads[0].get("covered_intervals", []):
        ax.axvspan(interval["start_s"], interval["end_s"], color="#86efac", alpha=0.8)
    ax.axvline(candidate.burst_time_s, color="#dc2626", linestyle="--", label="burst")
    ax.set(xlabel="time (s)", ylabel="coverage window", title="Q1 continuous coverage timeline")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "q1_coverage_timeline.png", dpi=180)
    plt.close(fig)

    times = np.linspace(detection[0].start_s, detection[-1].end_s, 400)
    margins = [
        single_smoke_gap(
            problem.ship.position(float(t)),
            candidate.smoke.burst_center_m,
            candidate.smoke.radius(float(t)),
            ship_radius_m=problem.constants.ship.effective_radius_m,
        )
        for t in times
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, margins, color="#1d4ed8", label="coverage gap (m)")
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set(xlabel="time (s)", ylabel="gap (m)", title="Q1 coverage margin curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "q1_margin_curve.png", dpi=180)
    plt.close(fig)

    ship, missile = _integrate_scenario(problem.scenario, problem.constants)
    trajectory_end = min(missile.end_time_s, detection[-1].end_s)
    trajectory_times = np.linspace(0.0, trajectory_end, 500)
    ship_positions = np.asarray([ship.position(float(t)) for t in trajectory_times])
    missile_positions = np.asarray([missile.position(float(t)) for t in trajectory_times])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ship_positions[:, 0], ship_positions[:, 1], label="ship", color="#2563eb")
    ax.plot(
        missile_positions[:, 0],
        missile_positions[:, 1],
        label="missile",
        color="#dc2626",
    )
    for segment in candidate.path.segments:
        ax.plot(
            [segment.start_position_m[0], segment.end_position_m[0]],
            [segment.start_position_m[1], segment.end_position_m[1]],
            color="#16a34a",
            linewidth=2,
        )
    ax.scatter(*problem.ship.position(0.0), color="#111827", s=24, label="ship start")
    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title="Q1 ship/missile trajectories and UAV path",
        aspect="equal",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "q1_trajectories.png", dpi=180)
    plt.close(fig)


def _write_counterfactual_figure(output: dict) -> None:
    formal = output["scenarios"][:4]
    counterfactual = output["counterfactual"]["representative_scenarios"]
    if not counterfactual:
        return
    labels = [item["scenario_id"].replace("q1_rebuild_", "") for item in counterfactual]
    coverage = [item["covered_duration_s"] for item in formal[: len(labels)]]
    separation = [item["minimum_separation_m"] for item in counterfactual]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, coverage, width=0.4, label="formal covered duration (s)")
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, separation, marker="o", color="#dc2626", label="lost minimum separation (m)")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("formal coverage (s)")
    ax2.set_ylabel("counterfactual separation (m)")
    ax.set_title("Formal baseline versus lost-guidance counterfactual")
    fig.tight_layout()
    fig.savefig(FIGURES / "q1_counterfactual_comparison.png", dpi=180)
    plt.close(fig)


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    scenarios = generate_q1_rebuild_matrix()
    benchmark_rows = []
    representative = scenarios[:4]
    method_scores = {method: [] for method in Q1_METHODS}
    for scenario in representative:
        problem = build_q1_problem(scenario)
        started = time.perf_counter()
        results = benchmark_q1_methods(problem, seed=20260731, evaluation_budget=24)
        runtime = (time.perf_counter() - started) / len(results)
        for result in results:
            row = _method_payload(result, runtime)
            benchmark_rows.append({"scenario_id": scenario.scenario_id, **row})
            if result.verification:
                method_scores[result.method].append(q1_verification_rank(result.verification))
    winner = max(
        Q1_METHODS,
        key=lambda method: tuple(np.mean(method_scores[method], axis=0))
        if method_scores[method]
        else (-np.inf,),
    )

    scenario_rows = []
    first_problem = None
    first_result = None
    for scenario in scenarios:
        problem = build_q1_problem(scenario)
        result = next(
            item
            for item in benchmark_q1_methods(problem, seed=20260731, evaluation_budget=32)
            if item.method == winner
        )
        payload = _scenario_payload(scenario, result, winner)
        scenario_rows.append(payload)
        if first_problem is None:
            first_problem = problem
            first_result = result

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "formal_baseline": True,
        "formal_baseline_model": "instantaneous_pure_pursuit",
        "matrix_shape": "4x4",
        "scenario_count": len(scenario_rows),
        "solver_verifier_separated": True,
        "objective_hierarchy": [
            "strict_full_window_coverage",
            "max_certified_total_coverage_duration",
            "min_max_continuous_exposure",
            "max_min_coverage_margin",
            "min_travel_or_resources",
        ],
        "algorithm_benchmark": benchmark_rows,
        "selected_method": winner,
        "scenarios": scenario_rows,
        "counterfactual": {
            "label": "experimental_counterfactual",
            "formal_baseline": False,
            "representative_scenarios": [],
        },
    }
    for scenario, _row in zip(scenarios[:4], scenario_rows[:4], strict=True):
        problem = build_q1_problem(scenario)
        candidate = next(
            item
            for item in benchmark_q1_methods(problem, seed=20260731, evaluation_budget=16)
            if item.method == winner
        ).best_candidate
        if candidate is None:
            continue
        counterfactual = simulate_lost_counterfactual(
            problem,
            candidate,
            LostCounterfactualParameters(tau_t_s=0.5, tau_l_s=5.0, t_r_s=1.0),
        )
        output["counterfactual"]["representative_scenarios"].append(
            {
                "scenario_id": scenario.scenario_id,
                "lost": counterfactual.lost,
                "reacquired": counterfactual.reacquired,
                "hit": counterfactual.hit,
                "minimum_separation_m": counterfactual.minimum_separation_m,
                "parameters": counterfactual.parameters,
            }
        )
    (RESULTS / "q1_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "scenario_id",
        "method",
        "verification_status",
        "covered_duration_s",
        "exposed_duration_s",
        "maximum_exposure_s",
        "minimum_margin_m",
        "flight_distance_m",
        "command_time_s",
        "drop_time_s",
        "burst_time_s",
    ]
    with (RESULTS / "q1_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in scenario_rows:
            writer.writerow({field: row.get(field) for field in fields})
    benchmark_fields = [
        "scenario_id",
        "method",
        "seed",
        "evaluation_budget",
        "evaluations",
        "runtime_s",
        "native_success",
        "native_status",
        "verification_status",
        "covered_duration_s",
        "maximum_exposure_s",
    ]
    with (RESULTS / "q1_algorithm_benchmark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=benchmark_fields)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    if first_problem and first_result:
        candidate = first_result.best_candidate
        _write_figures(first_problem, candidate, scenario_rows)
        _write_counterfactual_figure(output)
    return output


def check() -> int:
    required = (
        RESULTS / "q1_results.json",
        RESULTS / "q1_results.csv",
        RESULTS / "q1_algorithm_benchmark.csv",
        FIGURES / "q1_coverage_timeline.png",
        FIGURES / "q1_margin_curve.png",
        FIGURES / "q1_trajectories.png",
        FIGURES / "q1_counterfactual_comparison.png",
        ROOT / "docs/q1/q1-model.md",
        ROOT / "docs/q1/q1-algorithm-benchmark.md",
        ROOT / "docs/q1/q1-verification.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing Q1 rebuild artifacts: " + ", ".join(missing))
    payload = json.loads((RESULTS / "q1_results.json").read_text(encoding="utf-8"))
    if payload.get("scenario_count") != 16 or payload.get("formal_baseline") is not True:
        raise SystemExit("Q1 rebuild artifact contract is stale")
    print("Q1 rebuild artifacts are current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    run()
    print("wrote Q1 rebuild artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
