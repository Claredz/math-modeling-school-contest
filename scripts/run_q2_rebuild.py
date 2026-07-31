"""Run the bounded, event-parameterized Q2 candidate workflow."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from smoke_defense.q1_rebuild import build_q1_problem
from smoke_defense.q2_rebuild import (
    Q2CandidateResult,
    Q2CertificationStatus,
    Q2SolveResult,
    solve_q2_candidates,
)
from smoke_defense.scenario_matrix import generate_q1_rebuild_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "q2_rebuild"
FIGURES = ROOT / "figures" / "q2_rebuild"
Q1_RESULTS = ROOT / "results" / "q1_rebuild" / "q1_results.json"


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _status_rank(status: Q2CertificationStatus) -> float:
    return {
        Q2CertificationStatus.CERTIFIED_FEASIBLE: 2.0,
        Q2CertificationStatus.UNRESOLVED: 1.0,
        Q2CertificationStatus.CERTIFIED_INFEASIBLE: 0.0,
    }[status]


def _candidate_payload(candidate: Q2CandidateResult) -> dict:
    certificate = candidate.certificate
    return {
        "burst_times_s": list(candidate.burst_times_s),
        "center_times_s": list(candidate.center_times_s),
        "bomb_count": len(candidate.burst_times_s),
        "verification_status": certificate.status.value,
        "coverage_lower_s": certificate.coverage_lower_s,
        "coverage_upper_s": certificate.coverage_upper_s,
        "total_exposure_lower_s": certificate.total_exposure_lower_s,
        "total_exposure_upper_s": certificate.total_exposure_upper_s,
        "maximum_continuous_exposure_s": certificate.maximum_continuous_exposure_s,
        "joint_gain_s": certificate.joint_gain_s,
        "witness_time_s": certificate.witness_time_s,
        "unresolved_intervals": [
            {"start_s": item.start_s, "end_s": item.end_s}
            for item in certificate.unresolved_intervals
        ],
        "reason": certificate.reason,
        "solver_native_success": candidate.solver_native_success,
    }


def _load_q1_warm_starts() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    if not Q1_RESULTS.exists():
        return {}
    payload = json.loads(Q1_RESULTS.read_text(encoding="utf-8"))
    return {
        item["scenario_id"]: (
            (float(item["burst_time_s"]),),
            (float(item["center_time_s"]),),
        )
        for item in payload.get("scenarios", [])
        if item.get("burst_time_s") is not None
    }


def _best_multi(result: Q2SolveResult) -> Q2CandidateResult:
    multi = [item for item in result.candidates if len(item.burst_times_s) >= 2]
    return max(
        multi,
        key=lambda item: (
            _status_rank(item.certificate.status),
            item.certificate.coverage_lower_s,
            -item.certificate.maximum_continuous_exposure_s,
            item.certificate.joint_gain_s,
        ),
    )


def _write_figures(first_payload: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    q1_coverage = float(first_payload.get("q1_covered_duration_s", 0.0))
    q2_lower = float(first_payload["coverage_lower_s"])
    q2_upper = float(first_payload["coverage_upper_s"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Q1 single smoke", "Q2 multi-bomb lower", "Q2 multi-bomb upper"],
           [q1_coverage, q2_lower, q2_upper], color=["#94a3b8", "#2563eb", "#93c5fd"])
    ax.set_ylabel("certified coverage duration (s)")
    ax.set_title("Q2 joint-coverage bounds versus Q1")
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_joint_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    detection = first_payload["detection_components"]
    for interval in detection:
        ax.axvspan(interval["start_s"], interval["end_s"], color="#dbeafe", alpha=0.8)
    for burst in first_payload["burst_times_s"]:
        ax.axvline(burst, color="#dc2626", linestyle="--")
    ax.set(xlabel="time (s)", ylabel="detection window", title="Q2 event timeline")
    fig.tight_layout()
    fig.savefig(FIGURES / "q2_coverage_timeline.png", dpi=180)
    plt.close(fig)


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    scenarios = generate_q1_rebuild_matrix()[:4]
    q1_warm_starts = _load_q1_warm_starts()
    rows: list[dict] = []
    started_all = time.perf_counter()
    for scenario in scenarios:
        problem = build_q1_problem(scenario)
        started = time.perf_counter()
        warm_bursts, warm_centers = q1_warm_starts.get(
            scenario.scenario_id, ((), ())
        )
        solved = solve_q2_candidates(
            problem,
            warm_burst_times_s=warm_bursts,
            warm_center_times_s=warm_centers,
            maximum_candidates=40,
            polish=True,
        )
        runtime = time.perf_counter() - started
        selected = _best_multi(solved)
        q1_coverage = 0.0
        if Q1_RESULTS.exists():
            q1_payload = json.loads(Q1_RESULTS.read_text(encoding="utf-8"))
            q1_coverage = next(
                (
                    float(item.get("covered_duration_s", 0.0))
                    for item in q1_payload.get("scenarios", [])
                    if item.get("scenario_id") == scenario.scenario_id
                ),
            )
        row = _candidate_payload(selected)
        row.update(
            {
                "scenario_id": scenario.scenario_id,
                "runtime_s": runtime,
                "candidate_count": len(solved.candidates),
                "warm_start_count": solved.warm_start_count,
                "polish_success": solved.polish_success,
                "polish_status": solved.polish_status,
                "q1_covered_duration_s": q1_coverage,
                "relative_q1_improvement_lower_s": row["coverage_lower_s"] - q1_coverage,
                "detection_components": [
                    {"start_s": item.start_s, "end_s": item.end_s}
                    for item in problem.detection.components
                ],
            }
        )
        rows.append(row)
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "formal_baseline": True,
        "solver_verifier_separated": True,
        "search_scope": (
            "bounded event-parameterized candidates on four representative Q1 scenarios"
        ),
        "global_optimum_claim": False,
        "candidate_generation": "Q1 warm starts plus event anchors and 1-3 bomb combinations",
        "continuous_refinement": "SLSQP polish with one-second drop-spacing constraints",
        "joint_verification": "continuous time subdivision with conservative spatial envelopes",
        "unresolved_policy": "retain unresolved intervals and never promote them to feasible",
        "scenario_count": len(rows),
        "total_runtime_s": time.perf_counter() - started_all,
        "scenarios": rows,
    }
    (RESULTS / "q2_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "scenario_id", "bomb_count", "verification_status", "burst_times_s",
        "coverage_lower_s", "coverage_upper_s", "total_exposure_lower_s",
        "total_exposure_upper_s", "maximum_continuous_exposure_s", "joint_gain_s",
        "q1_covered_duration_s", "relative_q1_improvement_lower_s", "runtime_s",
    ]
    with (RESULTS / "q2_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    if rows:
        _write_figures(rows[0])
    return output


def check() -> int:
    required = (
        RESULTS / "q2_results.json",
        RESULTS / "q2_results.csv",
        FIGURES / "q2_joint_gain.png",
        FIGURES / "q2_coverage_timeline.png",
        ROOT / "docs/q2/q2-model.md",
        ROOT / "docs/q2/q2-algorithm.md",
        ROOT / "docs/q2/q2-verification.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing Q2 rebuild artifacts: " + ", ".join(missing))
    payload = json.loads((RESULTS / "q2_results.json").read_text(encoding="utf-8"))
    if payload.get("scenario_count") != 4 or payload.get("global_optimum_claim") is not False:
        raise SystemExit("Q2 rebuild artifact contract is stale")
    if any("verification_status" not in row for row in payload.get("scenarios", [])):
        raise SystemExit("Q2 verification status missing")
    print("Q2 rebuild artifacts are current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    run()
    print("wrote Q2 rebuild artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
