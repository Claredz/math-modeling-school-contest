"""Produce Q4 causal rolling scheduling artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from smoke_defense.q4_rebuild import (
    build_q4_tasks,
    schedule_causal_greedy,
    schedule_causal_rolling,
    schedule_offline_hindsight,
)

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
        rolling = schedule_causal_rolling(
            build_q4_tasks(), resource_capacity=capacity, horizon_end_s=30.0
        )
        greedy = schedule_causal_greedy(
            build_q4_tasks(), resource_capacity=capacity, horizon_end_s=30.0
        )
        hindsight = schedule_offline_hindsight(
            build_q4_tasks(), resource_capacity=capacity, horizon_end_s=30.0
        )
        rows.append(
            {
                "resource_case": label,
                "capacity": capacity,
                "status": rolling.status,
                "causal": rolling.causal,
                "total_value": rolling.total_value,
                "certified_value": rolling.certified_value,
                "rolling_value": rolling.certified_value,
                "greedy_value": greedy.certified_value,
                "hindsight_upper_bound": hindsight.certified_value,
                "rolling_vs_greedy_gap": rolling.certified_value - greedy.certified_value,
                "hindsight_gap": hindsight.certified_value - rolling.certified_value,
                "greedy_unresolved_task_ids": list(greedy.unresolved_task_ids),
                "hindsight_decision_task_ids": [
                    item.task_id for item in hindsight.decisions
                ],
                "decisions": [
                    {
                        "time_s": item.time_s,
                        "task_id": item.task_id,
                        "resources_used": item.resources_used,
                        "value": item.value,
                    }
                    for item in rolling.decisions
                ],
                "unresolved_task_ids": list(rolling.unresolved_task_ids),
                "reason": rolling.reason,
            }
        )
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "formal_baseline": True,
        "causal_rolling": True,
        "causal_greedy_baseline": True,
        "offline_hindsight_upper_bound": True,
        "task_package_source": "Q1-Q3 certified task-package interface",
        "global_optimum_claim": False,
        "resource_cases": rows,
    }
    (RESULTS / "q4_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "resource_case",
        "capacity",
        "status",
        "rolling_value",
        "greedy_value",
        "hindsight_upper_bound",
        "rolling_vs_greedy_gap",
        "hindsight_gap",
        "causal",
    ]
    with (RESULTS / "q4_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(rows)))
    width = 0.24
    ax.bar(
        [value - width for value in x],
        [item["rolling_value"] for item in rows],
        width=width,
        label="causal rolling",
        color="#16a34a",
    )
    ax.bar(
        x,
        [item["greedy_value"] for item in rows],
        width=width,
        label="causal greedy",
        color="#f59e0b",
    )
    ax.bar(
        [value + width for value in x],
        [item["hindsight_upper_bound"] for item in rows],
        width=width,
        label="offline hindsight upper bound",
        color="#94a3b8",
    )
    ax.set_xticks(x, [item["resource_case"] for item in rows])
    ax.set_ylabel("certified scheduled value")
    ax.set_title("Q4 causal rolling allocation under resource cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "q4_resource_cases.png", dpi=180)
    plt.close(fig)
    return output


def check() -> int:
    required = (
        RESULTS / "q4_results.json",
        RESULTS / "q4_results.csv",
        FIGURES / "q4_resource_cases.png",
        ROOT / "docs/q4/q4-model.md",
        ROOT / "docs/q4/q4-algorithm.md",
        ROOT / "docs/q4/q4-verification.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing Q4 rebuild artifacts: " + ", ".join(missing))
    payload = json.loads((RESULTS / "q4_results.json").read_text(encoding="utf-8"))
    if (
        payload.get("causal_rolling") is not True
        or payload.get("causal_greedy_baseline") is not True
        or payload.get("offline_hindsight_upper_bound") is not True
        or len(payload.get("resource_cases", [])) != 3
    ):
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
