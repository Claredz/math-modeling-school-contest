from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "sensitivity_rebuild"


def test_sensitivity_artifact_covers_required_categories():
    payload = json.loads(
        (RESULTS / "sensitivity_results.json").read_text(encoding="utf-8")
    )
    assert set(payload["categories"]) == {
        "missile_model",
        "lost_counterfactual",
        "smoke_drift",
        "response_delay",
        "uav_reachability",
    }
    assert payload["row_count"] >= 20
    assert (RESULTS / "sensitivity_results.csv").exists()
    figure = (
        RESULTS.parent.parent
        / "figures"
        / "sensitivity_rebuild"
        / "sensitivity_summary.png"
    )
    assert figure.exists()
