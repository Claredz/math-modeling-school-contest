"""Generate the minimum Stage 6 local sensitivity evidence."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from smoke_defense.coverage import single_smoke_gap
from smoke_defense.lost_counterfactual import (
    LostCounterfactualParameters,
    simulate_lost_counterfactual,
)
from smoke_defense.path_constraints import certify_operation_radius
from smoke_defense.q1 import solve_q1_scenario
from smoke_defense.q1_rebuild import build_q1_problem, construct_q1_candidate
from smoke_defense.scenario_matrix import (
    generate_q1_q3_matrix,
    generate_q1_rebuild_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sensitivity_rebuild"
FIGURES = ROOT / "figures" / "sensitivity_rebuild"
Q1_RESULTS = ROOT / "results" / "q1_rebuild" / "q1_results.json"


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _q1_reference():
    scenario = generate_q1_rebuild_matrix()[0]
    problem = build_q1_problem(scenario)
    payload = json.loads(Q1_RESULTS.read_text(encoding="utf-8"))
    row = next(item for item in payload["scenarios"] if item["scenario_id"] == scenario.scenario_id)
    candidate = construct_q1_candidate(
        problem,
        burst_time_s=float(row["burst_time_s"]),
        center_time_s=float(row["center_time_s"]),
    )
    return scenario, problem, candidate


def _row(
    category: str,
    parameter: str,
    value: float,
    metric: str,
    result: float | str | bool,
) -> dict:
    return {
        "category": category,
        "parameter": parameter,
        "value": value,
        "metric": metric,
        "result": result,
    }


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    _scenario, problem, candidate = _q1_reference()
    rows: list[dict] = []
    detection_start = problem.detection.components[0].start_s
    detection_end = problem.detection.components[-1].end_s

    # Missile guidance parameter sensitivity: formal IPP versus inertial rates.
    rows.append(
        _row(
            "missile_model",
            "instantaneous_pure_pursuit",
            0.0,
            "detection_duration_s",
            problem.detection.duration_s,
        )
    )
    inertial_scenarios = [
        item
        for item in generate_q1_q3_matrix()
        if item.scenario_id.startswith("q1_q3_front_d8000_")
        and item.missiles[0].max_turn_rate_deg_s == 10.0
    ]
    for inertial in inertial_scenarios:
        solved = solve_q1_scenario(inertial)
        rate = inertial.missiles[0].heading_response_rate_per_s or 0.0
        rows.append(
            _row(
                "missile_model",
                "inertial_heading_response_rate_per_s",
                float(rate),
                "detection_duration_s",
                solved.detection.duration_s,
            )
        )
    rows.append(
        _row(
            "missile_model",
            "proportional_navigation",
            0.0,
            "status",
            "deferred_no_navigation_ratio_or_overload_source",
        )
    )

    # Lost/reacquisition parameters are explicitly experimental only.
    for tau_t in (0.25, 0.5, 1.0):
        for tau_l in (3.0, 5.0, 8.0):
            for t_r in (0.5, 1.0, 2.0):
                result = simulate_lost_counterfactual(
                    problem,
                    candidate,
                    LostCounterfactualParameters(
                        tau_t_s=tau_t,
                        tau_l_s=tau_l,
                        t_r_s=t_r,
                    ),
                )
                rows.append(
                    _row(
                        "lost_counterfactual",
                        "tau_T_tau_L_T_R",
                        tau_t,
                        "minimum_separation_m",
                        result.minimum_separation_m,
                    )
                )

    # Bounded static drift proxy and smoke-radius margin sensitivity.
    sample_times = np.linspace(detection_start, detection_end, 240)
    for drift_bound in (0.0, 10.0, 20.0, 40.0):
        shifted_center = candidate.smoke.burst_center_m + np.array([0.0, drift_bound])
        minimum_gap = min(
            single_smoke_gap(
                problem.ship.position(float(time_s)),
                shifted_center,
                candidate.smoke.radius(float(time_s)),
                ship_radius_m=problem.constants.ship.effective_radius_m,
            )
            for time_s in sample_times
        )
        rows.append(
            _row(
                "smoke_drift",
                "bounded_drift_m",
                drift_bound,
                "minimum_gap_m",
                minimum_gap,
            )
        )

    # Response delay sensitivity is a causal-availability calculation; the
    # fixed-centre geometry is unchanged, while the earliest legal burst moves.
    for response_delay in (1.5, 2.0, 2.5, 3.0):
        earliest_burst = response_delay + 3.5
        rows.append(
            _row(
                "response_delay",
                "command_to_drop_s",
                response_delay,
                "available_detection_after_earliest_burst_s",
                max(0.0, detection_end - max(detection_start, earliest_burst)),
            )
        )

    # UAV reachability sensitivity keeps the path fixed and varies only the
    # allowed operation radius, which is a transparent local robustness check.
    radius_certificate = certify_operation_radius(
        candidate.path,
        operation_radius_m=problem.constants.uav.operation_radius_m,
    )
    max_distance = radius_certificate.maximum_value or 0.0
    for radius in (8000.0, 10000.0, 12000.0, 14000.0):
        rows.append(
            _row(
                "uav_reachability",
                "operation_radius_m",
                radius,
                "path_feasible",
                max_distance <= radius + 1e-9,
            )
        )

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "formal_baseline": True,
        "experimental_counterfactual": {
            "lost_guidance": True,
            "formal_baseline": False,
            "parameters": ["tau_T_s", "tau_L_s", "T_R_s"],
        },
        "categories": [
            "missile_model",
            "lost_counterfactual",
            "smoke_drift",
            "response_delay",
            "uav_reachability",
        ],
        "row_count": len(rows),
        "rows": rows,
    }
    (RESULTS / "sensitivity_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = ["category", "parameter", "value", "metric", "result"]
    with (RESULTS / "sensitivity_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    drift = [item for item in rows if item["category"] == "smoke_drift"]
    axes[0, 0].plot(
        [item["value"] for item in drift],
        [item["result"] for item in drift],
        marker="o",
    )
    axes[0, 0].set(xlabel="bounded drift (m)", ylabel="minimum gap (m)")
    response = [item for item in rows if item["category"] == "response_delay"]
    axes[0, 1].plot(
        [item["value"] for item in response],
        [item["result"] for item in response],
        marker="o",
    )
    axes[0, 1].set(xlabel="response delay (s)", ylabel="available window (s)")
    reach = [item for item in rows if item["category"] == "uav_reachability"]
    axes[1, 0].plot(
        [item["value"] for item in reach],
        [float(item["result"]) for item in reach],
        marker="o",
    )
    axes[1, 0].set(xlabel="operation radius (m)", ylabel="path feasible (0/1)")
    inertial = [
        item
        for item in rows
        if item["category"] == "missile_model"
        and item["metric"] == "detection_duration_s"
    ]
    axes[1, 1].bar(
        [str(item["value"]) for item in inertial],
        [float(item["result"]) for item in inertial],
    )
    axes[1, 1].set(xlabel="guidance parameter / IPP", ylabel="detection duration (s)")
    fig.suptitle("Stage 6 local sensitivity evidence")
    fig.tight_layout()
    fig.savefig(FIGURES / "sensitivity_summary.png", dpi=180)
    plt.close(fig)
    return output


def check() -> int:
    required = (
        RESULTS / "sensitivity_results.json",
        RESULTS / "sensitivity_results.csv",
        FIGURES / "sensitivity_summary.png",
        ROOT / "docs/stage_06_sensitivity.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing sensitivity artifacts: " + ", ".join(missing))
    payload = json.loads((RESULTS / "sensitivity_results.json").read_text(encoding="utf-8"))
    if payload.get("row_count", 0) < 20 or len(payload.get("categories", [])) < 5:
        raise SystemExit("sensitivity artifact contract is stale")
    print("sensitivity artifacts are current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    run()
    print("wrote sensitivity artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
