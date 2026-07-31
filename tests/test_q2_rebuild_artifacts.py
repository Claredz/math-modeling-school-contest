from __future__ import annotations

import json
from pathlib import Path


def test_q2_artifact_contains_paths_and_bounds():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "results/q2_rebuild/q2_results.json").read_text(encoding="utf-8")
    )
    assert payload["global_optimum_claim"] is False
    assert payload["scenarios"]
    for row in payload["scenarios"]:
        assert row["uav_path"]
        assert row["coverage_upper_s"] >= row["coverage_lower_s"]
        assert "verification_status" in row
        assert row["joint_coverage_lower_s"] == row["coverage_lower_s"]
        assert row["joint_gain_s"] == (
            row["joint_coverage_lower_s"]
            - row["best_single_smoke_coverage_lower_s"]
        )
        assert row["maximum_exposure_lower_s"] <= row["maximum_exposure_upper_s"]
        assert row["maximum_continuous_exposure_s"] == row["maximum_exposure_lower_s"]
        assert "best_single_smoke_candidate_id" in row
