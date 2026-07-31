from __future__ import annotations

import json

from scripts.run_sensitivity_rebuild import RESULTS, run


def test_sensitivity_artifact_covers_required_categories():
    payload = run()
    assert set(payload["categories"]) == {
        "missile_model",
        "lost_counterfactual",
        "smoke_drift",
        "response_delay",
        "uav_reachability",
    }
    assert payload["row_count"] >= 20
    stored = json.loads((RESULTS / "sensitivity_results.json").read_text(encoding="utf-8"))
    assert stored["row_count"] == payload["row_count"]
