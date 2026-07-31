"""Produce the bounded Q3 cooperative reconstruction artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from smoke_defense.q1_rebuild import build_q1_problem
from smoke_defense.q3_rebuild import generate_q3_plan
from smoke_defense.scenario_matrix import generate_q1_rebuild_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "q3_rebuild"
FIGURES = ROOT / "figures" / "q3_rebuild"


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in generate_q1_rebuild_matrix()[:4]:
        problem = build_q1_problem(scenario)
        plan, certificate = generate_q3_plan(problem)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "interpretation": plan.interpretation,
                "uav_count": 3,
                "bomb_count": 3,
                "burst_times_s": list(plan.burst_times_s),
                "center_times_s": list(plan.center_times_s),
                "verification_status": certificate.status.value,
                "operation_radius_ok": certificate.operation_radius_ok,
                "coverage_lower_s": certificate.joint.coverage_lower_s,
                "coverage_upper_s": certificate.joint.coverage_upper_s,
                "total_exposure_lower_s": certificate.joint.total_exposure_lower_s,
                "total_exposure_upper_s": certificate.joint.total_exposure_upper_s,
                "maximum_continuous_exposure_s": certificate.joint.maximum_continuous_exposure_s,
                "joint_gain_s": certificate.joint.joint_gain_s,
                "unresolved_intervals": [
                    {"start_s": item.start_s, "end_s": item.end_s}
                    for item in certificate.joint.unresolved_intervals
                ],
                "reason": certificate.reason,
            }
        )
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "formal_baseline": True,
        "main_interpretation": "three_uavs_exactly_one_bomb_each",
        "relaxation": "at_most_one_bomb_per_uav_is_not_used_as_main_result",
        "solver_verifier_separated": True,
        "scenario_count": len(rows),
        "global_optimum_claim": False,
        "scenarios": rows,
    }
    (RESULTS / "q3_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        first = rows[0]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            list(range(1, 4)),
            first["burst_times_s"],
            marker="o",
            color="#7c3aed",
        )
        ax.set(xlabel="UAV index", ylabel="burst time (s)", title="Q3 cooperative release schedule")
        fig.tight_layout()
        fig.savefig(FIGURES / "q3_schedule.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            ["coverage lower", "coverage upper"],
            [first["coverage_lower_s"], first["coverage_upper_s"]],
            color=["#7c3aed", "#c4b5fd"],
        )
        ax.set_ylabel("duration (s)")
        ax.set_title("Q3 joint coverage certificate bounds")
        fig.tight_layout()
        fig.savefig(FIGURES / "q3_coverage_bounds.png", dpi=180)
        plt.close(fig)
    return output


def check() -> int:
    required = (
        RESULTS / "q3_results.json",
        FIGURES / "q3_schedule.png",
        FIGURES / "q3_coverage_bounds.png",
        ROOT / "docs/q3/q3-model.md",
        ROOT / "docs/q3/q3-algorithm.md",
        ROOT / "docs/q3/q3-verification.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing Q3 rebuild artifacts: " + ", ".join(missing))
    payload = json.loads((RESULTS / "q3_results.json").read_text(encoding="utf-8"))
    if payload.get("main_interpretation") != "three_uavs_exactly_one_bomb_each":
        raise SystemExit("Q3 interpretation is stale")
    print("Q3 rebuild artifacts are current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    run()
    print("wrote Q3 rebuild artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
