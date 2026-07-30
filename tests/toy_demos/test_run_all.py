from __future__ import annotations

import json
from pathlib import Path

from experiments.toy_demos import run_all

EXPECTED_FILES = {
    "q1_continuous_optimization.json": 5,
    "q2_constraint_generation.json": 1,
    "q2_joint_prototype.json": 2,
    "q3_multiobjective.json": 5,
    "q4_scheduling.json": 4,
}
RECORD_FIELDS = {
    "demo_name",
    "solver",
    "seed",
    "objective",
    "runtime_s",
    "converged",
    "passed_manual_case",
    "failure_reason",
    "metadata",
}


def test_build_artifacts_uses_standard_records_and_explicit_toy_boundary() -> None:
    artifacts = run_all.build_artifacts(seed=20260731)

    assert set(artifacts) == set(EXPECTED_FILES)
    for filename, expected_record_count in EXPECTED_FILES.items():
        artifact = artifacts[filename]
        assert artifact["schema_version"] == 1
        assert artifact["synthetic"] is True
        assert artifact["formal_result"] is False
        assert artifact["seed"] == 20260731
        assert isinstance(artifact["guarantee_boundary"], list)
        assert artifact["guarantee_boundary"]
        assert isinstance(artifact["module_summary"], dict)
        assert len(artifact["records"]) == expected_record_count
        assert all(set(record) == RECORD_FIELDS for record in artifact["records"])


def test_normalized_artifacts_are_deterministic_despite_runtime() -> None:
    first = run_all.build_artifacts(seed=20260731)
    second = run_all.build_artifacts(seed=20260731)

    assert run_all.normalize_for_check(first) == run_all.normalize_for_check(second)
    assert any(
        record["runtime_s"] >= 0.0
        for artifact in first.values()
        for record in artifact["records"]
    )


def test_write_and_check_detect_missing_or_stale_artifacts(tmp_path: Path) -> None:
    artifacts = run_all.build_artifacts(seed=20260731)
    written = run_all.write_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        artifacts=artifacts,
    )

    assert set(written) == set(EXPECTED_FILES)
    ok, issues = run_all.check_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        expected_artifacts=artifacts,
    )
    assert ok is True
    assert issues == ()

    missing = tmp_path / "q2_joint_prototype.json"
    missing.unlink()
    ok, issues = run_all.check_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        expected_artifacts=artifacts,
    )
    assert ok is False
    assert any("missing" in issue and missing.name in issue for issue in issues)

    run_all.write_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        artifacts=artifacts,
    )
    stale = tmp_path / "q4_scheduling.json"
    payload = json.loads(stale.read_text(encoding="utf-8"))
    payload["module_summary"]["offline_upper_bound"] = -1
    stale.write_text(json.dumps(payload), encoding="utf-8")
    ok, issues = run_all.check_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        expected_artifacts=artifacts,
    )
    assert ok is False
    assert any("stale" in issue and stale.name in issue for issue in issues)


def test_cli_check_returns_nonzero_for_missing_artifacts(tmp_path: Path) -> None:
    assert run_all.main(["--check", "--output-dir", str(tmp_path)]) == 1
