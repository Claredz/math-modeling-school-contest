from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_q1_rebuild_artifact_contract_exists():
    result_path = ROOT / "results" / "q1_rebuild" / "q1_results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["formal_baseline"] is True
    assert payload["matrix_shape"] == "4x4"
    assert payload["solver_verifier_separated"] is True
    assert payload["scenario_count"] == 16
    assert payload["algorithm_benchmark"]
    assert len(payload["scenarios"]) == 16
    assert {item["verification_status"] for item in payload["scenarios"]} <= {
        "certified_feasible",
        "certified_infeasible",
        "unresolved",
    }
    assert payload["counterfactual"]["formal_baseline"] is False


def test_q1_rebuild_has_paper_ready_docs_and_figures():
    for relative in (
        "docs/q1/q1-model.md",
        "docs/q1/q1-algorithm-benchmark.md",
        "docs/q1/q1-verification.md",
        "results/q1_rebuild/q1_results.csv",
        "figures/q1_rebuild/q1_coverage_timeline.png",
        "figures/q1_rebuild/q1_margin_curve.png",
    ):
        assert (ROOT / relative).exists(), relative
