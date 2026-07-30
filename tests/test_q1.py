import json

import pytest

import smoke_defense.q1 as q1_module
from smoke_defense.coverage import CertificationStatus
from smoke_defense.q1 import (
    solve_q1_guidance_sweep,
    write_q1_markdown_summary,
    write_q1_sweep_result,
)


@pytest.fixture(scope="module")
def front_sweep():
    return solve_q1_guidance_sweep(direction="front", distance_m=10000.0)


def test_q1_sweep_runs_nine_formal_models_and_one_ablation(front_sweep):
    assert len(front_sweep.formal_results) == 9
    assert front_sweep.ablation_result.model_layer == "ablation"
    assert all(result.best_candidate is not None for result in front_sweep.formal_results)


def test_q1_long_detection_component_is_reported_infeasible(front_sweep):
    reference = front_sweep.reference_result

    assert reference.strict_status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert reference.duration_certificate.status is CertificationStatus.CERTIFIED_INFEASIBLE
    assert reference.best_candidate.covered_duration_s <= 80.0 / 7.71 + 1e-6


def test_ablation_keeps_old_duration_bound_only_as_reference(front_sweep):
    duration = front_sweep.ablation_result.detection.duration_s

    assert 24.1677 <= duration <= 25.3610


def test_reference_candidate_is_cross_validated_on_all_nine_models(front_sweep):
    assert len(front_sweep.cross_validation) == 9
    assert isinstance(front_sweep.parameter_sensitive, bool)
    assert front_sweep.worst_case_scenario_id


def test_q1_result_writes_traceable_json(front_sweep, tmp_path):
    output = tmp_path / "q1.json"

    write_q1_sweep_result(
        (front_sweep,),
        output,
        git_sha="test-sha",
        random_seed=20260730,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["git_sha"] == "test-sha"
    assert payload["random_seed"] == 20260730
    assert payload["model_contract_version"] == "v0.2"
    formal_result = payload["sweeps"][0]["formal_results"][0]
    assert formal_result["scenario_hash"]
    assert formal_result["assumption_ids"]
    assert formal_result["assumption_ids"] == [
        f"A-{index:03d}" for index in range(1, 23)
    ]
    assert payload["assumption_register_version"] == "v0.3"
    assert formal_result["detection_events"]
    assert {"appearance", "hit"} <= {
        event["kind"] for event in formal_result["detection_events"]
    }


def test_q1_writes_reader_facing_markdown_summary(front_sweep, tmp_path):
    output = tmp_path / "README.md"

    write_q1_markdown_summary((front_sweep,), output)
    text = output.read_text(encoding="utf-8")

    assert "# Q1 惯性纯追踪计算结果" in text
    assert "front" in text
    assert "certified_infeasible" in text


@pytest.mark.parametrize(
    ("direction", "expected_sensitive"),
    [("front", False), ("side", True)],
)
def test_empty_cross_validation_still_uses_detection_sensitivity(
    monkeypatch,
    direction,
    expected_sensitive,
):
    monkeypatch.setattr(
        q1_module,
        "_cross_validate_reference",
        lambda *_args, **_kwargs: (),
    )

    sweep = solve_q1_guidance_sweep(
        direction=direction,
        distance_m=8000.0,
    )

    assert sweep.cross_validation == ()
    assert sweep.parameter_sensitive is expected_sensitive
    assert sweep.worst_case_scenario_id == sweep.reference_result.scenario_id
