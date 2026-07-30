from __future__ import annotations

import json
import math
import subprocess
import sys
from copy import deepcopy
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
        if filename != "q3_multiobjective.json":
            assert all(record["seed"] == artifact["seed"] for record in artifact["records"])
    q3_records = artifacts["q3_multiobjective.json"]["records"]
    q3_summary = artifacts["q3_multiobjective.json"]["module_summary"]
    assert q3_records[0]["seed"] == 20260731
    assert [record["seed"] for record in q3_records[1:]] == [
        20260731,
        20260732,
        20260733,
        20260734,
    ]
    assert q3_summary["base_seed"] == 20260731
    assert q3_summary["nsga2_seeds"] == [20260731, 20260732, 20260733, 20260734]


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
    assert any(
        "module_summary.offline_upper_bound" in issue
        for issue in issues
        if stale.name in issue
    )
    assert any(
        "actual=-1" in issue and "expected=19.0" in issue
        for issue in issues
        if stale.name in issue
    )


def test_cli_check_returns_nonzero_for_missing_artifacts(tmp_path: Path) -> None:
    assert run_all.main(["--check", "--output-dir", str(tmp_path)]) == 1


def test_check_tolerates_cross_platform_float_noise_but_detects_material_drift(
    tmp_path: Path,
) -> None:
    artifacts = run_all.build_artifacts(seed=20260731)
    run_all.write_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        artifacts=artifacts,
    )
    target = tmp_path / "q2_joint_prototype.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    baseline = payload["module_summary"]["routes"][0]["global_gap"]

    payload["module_summary"]["routes"][0]["global_gap"] = baseline + 1e-6
    target.write_text(json.dumps(payload), encoding="utf-8")
    ok, issues = run_all.check_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        expected_artifacts=artifacts,
    )
    assert ok is True
    assert issues == ()

    payload["module_summary"]["routes"][0]["global_gap"] = baseline + 1e-3
    target.write_text(json.dumps(payload), encoding="utf-8")
    ok, issues = run_all.check_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        expected_artifacts=artifacts,
    )
    assert ok is False
    assert any(
        "module_summary.routes[0].global_gap" in issue
        for issue in issues
        if target.name in issue
    )


def test_check_rejects_invalid_record_schema_before_ignoring_runtime(
    tmp_path: Path,
) -> None:
    artifacts = run_all.build_artifacts(seed=20260731)
    invalid_records = []

    runtime_string = deepcopy(artifacts)
    runtime_string["q1_continuous_optimization.json"]["records"][0]["runtime_s"] = "fast"
    invalid_records.append(runtime_string)

    missing_field = deepcopy(artifacts)
    missing_field["q2_constraint_generation.json"]["records"][0].pop("solver")
    invalid_records.append(missing_field)

    nan_objective = deepcopy(artifacts)
    nan_objective["q3_multiobjective.json"]["records"][0]["objective"] = math.nan
    invalid_records.append(nan_objective)

    negative_runtime = deepcopy(artifacts)
    negative_runtime["q4_scheduling.json"]["records"][0]["runtime_s"] = -0.1
    invalid_records.append(negative_runtime)

    for invalid in invalid_records:
        tmp_path.mkdir(parents=True, exist_ok=True)
        for filename, payload in invalid.items():
            (tmp_path / filename).write_text(
                json.dumps(payload, allow_nan=True),
                encoding="utf-8",
            )
        ok, issues = run_all.check_artifacts(
            output_dir=tmp_path,
            seed=20260731,
            expected_artifacts=artifacts,
        )
        assert ok is False
        assert any("invalid artifact" in issue for issue in issues)


def test_cli_check_returns_nonzero_for_invalid_top_level_schema(tmp_path: Path) -> None:
    artifacts = run_all.build_artifacts(seed=20260731)
    artifacts["q2_joint_prototype.json"]["unexpected"] = True
    run_all.write_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        artifacts=artifacts,
    )

    ok, issues = run_all.check_artifacts(
        output_dir=tmp_path,
        seed=20260731,
        expected_artifacts=run_all.build_artifacts(seed=20260731),
    )
    assert ok is False
    assert any("invalid artifact" in issue for issue in issues)
    assert run_all.main(["--check", "--output-dir", str(tmp_path)]) == 1


def test_committed_artifacts_are_current_from_module_and_direct_script() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    commands = (
        [sys.executable, "-m", "experiments.toy_demos.run_all", "--check"],
        [
            sys.executable,
            "experiments/toy_demos/run_all.py",
            "--check",
        ],
    )

    completed = [
        subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
        for command in commands
    ]

    assert completed[0].returncode == 0, completed[0].stdout + completed[0].stderr
    assert completed[1].returncode == 0, completed[1].stdout + completed[1].stderr
