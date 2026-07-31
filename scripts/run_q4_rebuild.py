"""Produce Q4 causal rolling scheduling artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from smoke_defense.q4_rebuild import build_q4_tasks, schedule_causal_rolling

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "q4_rebuild"
FIGURES = ROOT / "figures" / "q4_rebuild"


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, capacity in (("abundant", 4), ("critical", 2), ("shortage", 1)):
        certificate = schedule_causal_rolling(
            build_q4_tasks(), resource_capacity=capacity, horizon_end_s=30.0
        )
        rows.append(
            {
                "resource_case": label,
                "capacity": capacity,
                "status": certificate.status,
                "causal": certificate.causal,
                "total_value": certificate.total_value,
                "certified_value": certificate.certified_value,
                "decisions": [
                    {
                        "time_s": item.time_s,
                        "task_id": item.task_id,
                        "resources_used": item.resources_used,
                        "value": item.value,
                    }
                    for item in certificate.decisions
                ],
                "unresolved_task_ids": list(certificate.unresolved_task_ids),
                "reason": certificate.reason,
            }
        )
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "formal_baseline": True,
        "causal_rolling": True,
        "task_package_source": "Q1-Q3 certified task-package interface",
        "global_optimum_claim": False,
        "resource_cases": rows,
    }
    (RESULTS / "q4_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [item["resource_case"] for item in rows],
        [item["certified_value"] for item in rows],
        color=["#16a34a", "#f59e0b", "#dc2626"],
    )
    ax.set_ylabel("certified scheduled value")
    ax.set_title("Q4 causal rolling allocation under resource cases")
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_resource_cases.png", dpi=180)
    plt.close(fig)
    return output


def check() -> int:
    required = (
        RESULTS / "q4_results.json",
        FIGURES / "q4_resource_cases.png",
        ROOT / "docs/q4/q4-model.md",
        ROOT / "docs/q4/q4-algorithm.md",
        ROOT / "docs/q4/q4-verification.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing Q4 rebuild artifacts: " + ", ".join(missing))
    payload = json.loads((RESULTS / "q4_results.json").read_text(encoding="utf-8"))
    if payload.get("causal_rolling") is not True or len(payload.get("resource_cases", [])) != 3:
        raise SystemExit("Q4 artifact contract is stale")
    print("Q4 rebuild artifacts are current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    run()
    print("wrote Q4 rebuild artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
