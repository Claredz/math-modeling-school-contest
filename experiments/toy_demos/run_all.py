"""Run and audit every isolated synthetic model-selection demonstration."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from math import isclose
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from experiments.toy_demos.common import ToyRunRecord
from experiments.toy_demos.q1_continuous_optimization import run_demo as run_q1
from experiments.toy_demos.q2_constraint_generation import run_constraint_generation
from experiments.toy_demos.q2_joint_prototype import run_q2_joint_demo
from experiments.toy_demos.q3_multiobjective import (
    assess_nsga2,
    exact_pareto_front,
    solve_epsilon,
)
from experiments.toy_demos.q4_scheduling import run_demo as run_q4

DEFAULT_SEED = 20260731
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_FLOAT_CHECK_ABS_TOL = 1e-12
SOLVER_METRIC_CHECK_ABS_TOL = 1e-5
SOLVER_COORDINATE_CHECK_ABS_TOL = 1e-4
FLOAT_CHECK_REL_TOL = 1e-12
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "module",
    "synthetic",
    "formal_result",
    "seed",
    "guarantee_boundary",
    "records",
    "module_summary",
}
_RECORD_FIELDS = {
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
_EXPECTED_RECORD_COUNTS = {
    "q1_continuous_optimization.json": 5,
    "q2_constraint_generation.json": 1,
    "q2_joint_prototype.json": 2,
    "q3_multiobjective.json": 5,
    "q4_scheduling.json": 4,
}
_EXPECTED_TOTAL_RECORDS = 17
_BOUNDARY = [
    "All instances and parameters in this directory are synthetic.",
    "These records compare algorithm behavior; they are not contest answers.",
    "A converged toy run is not a proof that the corresponding formal model is valid.",
]


def _artifact(
    *,
    module: str,
    seed: int,
    records: Sequence[ToyRunRecord],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "module": module,
        "synthetic": True,
        "formal_result": False,
        "seed": seed,
        "guarantee_boundary": list(_BOUNDARY),
        "records": [record.to_dict() for record in records],
        "module_summary": dict(summary),
    }


def _q1_artifact(seed: int) -> dict[str, Any]:
    results = run_q1(seed=seed)
    records = tuple(result.to_toy_record() for result in results.values())
    return _artifact(
        module="q1_continuous_optimization",
        seed=seed,
        records=records,
        summary={
            "method_count": len(results),
            "common_evaluation_budget": next(iter(results.values())).budget,
            "verified_methods": [
                name for name, result in results.items() if result.verified
            ],
            "unresolved_methods": [
                name for name, result in results.items() if not result.solver_success
            ],
            "known_synthetic_upper_bound": 10.0,
        },
    )


def _q2_constraint_artifact(seed: int) -> dict[str, Any]:
    started_at = perf_counter()
    result = run_constraint_generation(seed=seed)
    runtime_s = perf_counter() - started_at
    record = result.to_toy_record(runtime_s=runtime_s)
    return _artifact(
        module="q2_constraint_generation",
        seed=seed,
        records=(record,),
        summary={
            "initial_finite_master_misses_violation": result.current_violations[0] > 0.0,
            "oracle_witnesses_added": [list(point) for point in result.added_witnesses],
            "final_violation": result.current_violations[-1],
            "converged": result.converged,
            "failure_reason": result.failure_reason,
        },
    )


def _route_summary(route: Any) -> dict[str, Any]:
    return {
        "route_name": route.route_name,
        "verified_objective": route.verified_objective,
        "converged": route.converged,
        "locally_converged": route.local_solver_converged,
        "globally_resolved": route.globally_resolved,
        "unresolved": route.unresolved,
        "global_lower_bound": route.global_lower_bound,
        "global_upper_bound": route.global_upper_bound,
        "global_gap": route.global_gap,
        "failure_reason": route.failure_reason,
    }


def _q2_joint_artifact(seed: int) -> dict[str, Any]:
    result = run_q2_joint_demo(seed=seed)
    routes = (result.candidate_route, result.oracle_route)
    return _artifact(
        module="q2_joint_prototype",
        seed=seed,
        records=tuple(route.record for route in routes),
        summary={
            "grid_baseline": {
                "lower_bound": result.grid_bounds.lower_bound,
                "global_upper_bound": result.grid_bounds.global_upper_bound,
                "grid_step": result.grid_bounds.grid_step,
                "evaluated_schedules": result.grid_bounds.evaluated_schedules,
                "bound_source": result.grid_bounds.bound_source,
            },
            "routes": [_route_summary(route) for route in routes],
            "unresolved_routes": [
                route.route_name for route in routes if route.unresolved
            ],
        },
    )


def _q3_artifact(seed: int) -> dict[str, Any]:
    exact_front = exact_pareto_front()
    epsilon = solve_epsilon(risk_limit=8.0, seed=seed)
    nsga_seeds = (seed, seed + 1, seed + 2, seed + 3)
    assessment = assess_nsga2(seeds=nsga_seeds)
    records = (epsilon.record, *(run.record for run in assessment.runs))
    return _artifact(
        module="q3_multiobjective",
        seed=seed,
        records=records,
        summary={
            "base_seed": seed,
            "exact_pareto_front": [
                {"code": item.code, "benefit": item.benefit, "risk": item.risk}
                for item in exact_front
            ],
            "epsilon_selected_code": (
                epsilon.portfolio.code if epsilon.portfolio is not None else None
            ),
            "epsilon_on_exact_front": epsilon.portfolio in exact_front,
            "nsga2_seeds": list(nsga_seeds),
            "nsga2_mean_coverage": assessment.mean_coverage,
            "nsga2_minimum_coverage": assessment.minimum_coverage,
            "nsga2_mean_precision": assessment.mean_precision,
            "nsga2_mean_jaccard": assessment.mean_jaccard,
            "nsga2_total_false_positives": assessment.total_false_positives,
        },
    )


def _q4_artifact(seed: int) -> dict[str, Any]:
    results = run_q4(seed=seed)
    offline = results["offline_milp"]
    return _artifact(
        module="q4_scheduling",
        seed=seed,
        records=tuple(result.record for result in results.values()),
        summary={
            "offline_upper_bound": offline.objective,
            "policies": {
                name: {
                    "selected_ids": list(result.selected_ids),
                    "objective": result.objective,
                    "converged": result.converged,
                    "unresolved": result.unresolved,
                    "verified": result.verified,
                    "failure_reason": result.failure,
                }
                for name, result in results.items()
            },
            "offline_is_hindsight_only": True,
            "causal_policies_do_not_see_future_releases": True,
        },
    )


def build_artifacts(*, seed: int = DEFAULT_SEED) -> dict[str, dict[str, Any]]:
    """Execute all five synthetic modules and return their JSON payloads."""

    return {
        "q1_continuous_optimization.json": _q1_artifact(seed),
        "q2_constraint_generation.json": _q2_constraint_artifact(seed),
        "q2_joint_prototype.json": _q2_joint_artifact(seed),
        "q3_multiobjective.json": _q3_artifact(seed),
        "q4_scheduling.json": _q4_artifact(seed),
    }


def normalize_for_check(value: Any) -> Any:
    """Return a deep deterministic view with wall-clock runtimes ignored."""

    normalized = copy.deepcopy(value)

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "runtime_s":
                    item[key] = "<ignored:wall-clock>"
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(normalized)
    return normalized


def _difference_descriptions(
    actual: Any,
    expected: Any,
    *,
    prefix: str = "",
) -> tuple[str, ...]:
    """Return stable leaf-level descriptions of normalized artifact differences."""

    if isinstance(actual, dict) and isinstance(expected, dict):
        paths: list[str] = []
        for key in sorted(set(actual) | set(expected)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in actual or key not in expected:
                actual_value = actual.get(key, "<missing>")
                expected_value = expected.get(key, "<missing>")
                paths.append(
                    f"{path} (actual={actual_value!r}, expected={expected_value!r})"
                )
            else:
                paths.extend(
                    _difference_descriptions(actual[key], expected[key], prefix=path)
                )
        return tuple(paths)
    if isinstance(actual, list) and isinstance(expected, list):
        paths = []
        for index in range(max(len(actual), len(expected))):
            path = f"{prefix}[{index}]"
            if index >= len(actual) or index >= len(expected):
                actual_value = actual[index] if index < len(actual) else "<missing>"
                expected_value = expected[index] if index < len(expected) else "<missing>"
                paths.append(
                    f"{path} (actual={actual_value!r}, expected={expected_value!r})"
                )
            else:
                paths.extend(
                    _difference_descriptions(actual[index], expected[index], prefix=path)
                )
        return tuple(paths)
    if isinstance(actual, float) and isinstance(expected, float):
        if ".metadata.burst_times[" in prefix or ".metadata.master_values[" in prefix:
            absolute_tolerance = SOLVER_COORDINATE_CHECK_ABS_TOL
        elif prefix.endswith(
            (
                ".objective",
                ".global_gap",
                ".global_lower_bound",
                ".verified_objective",
            )
        ):
            absolute_tolerance = SOLVER_METRIC_CHECK_ABS_TOL
        else:
            absolute_tolerance = DEFAULT_FLOAT_CHECK_ABS_TOL
        if isclose(
            actual,
            expected,
            rel_tol=FLOAT_CHECK_REL_TOL,
            abs_tol=absolute_tolerance,
        ):
            return ()
    elif actual == expected:
        return ()
    path = prefix or "<root>"
    return (f"{path} (actual={actual!r}, expected={expected!r})",)


def _json_text(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _validate_artifact_payload(
    filename: str,
    payload: Any,
    *,
    seed: int,
) -> int:
    if not isinstance(payload, dict):
        raise TypeError("top level must be an object")
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("top-level fields do not match the artifact contract")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    expected_module = filename.removesuffix(".json")
    if payload["module"] != expected_module:
        raise ValueError(f"module must be {expected_module!r}")
    if payload["synthetic"] is not True or payload["formal_result"] is not False:
        raise ValueError("artifact must be explicitly synthetic and non-formal")
    if type(payload["seed"]) is not int or payload["seed"] != seed:
        raise ValueError("top-level seed must match the requested integer seed")
    boundary = payload["guarantee_boundary"]
    if (
        not isinstance(boundary, list)
        or not boundary
        or any(not isinstance(item, str) or not item.strip() for item in boundary)
    ):
        raise ValueError("guarantee_boundary must be a nonempty list of nonempty strings")
    if not isinstance(payload["module_summary"], dict):
        raise TypeError("module_summary must be an object")
    json.dumps(payload["module_summary"], allow_nan=False)

    records = payload["records"]
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    expected_count = _EXPECTED_RECORD_COUNTS.get(filename)
    if expected_count is None or len(records) != expected_count:
        raise ValueError(f"record count must be {expected_count}")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise ValueError(f"record {index} fields do not match ToyRunRecord")
        reconstructed = ToyRunRecord(**record)
        if filename != "q3_multiobjective.json" and reconstructed.seed != seed:
            raise ValueError(f"record {index} seed does not match the artifact seed")
        if filename == "q3_multiobjective.json" and index == 0 and reconstructed.seed != seed:
            raise ValueError("Q3 epsilon record seed does not match the base seed")
        original_json = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if reconstructed.to_json() != original_json:
            raise ValueError(f"record {index} is not in normalized ToyRunRecord form")
    if filename == "q3_multiobjective.json":
        summary = payload["module_summary"]
        derived_seeds = [record["seed"] for record in records[1:]]
        if summary.get("base_seed") != seed:
            raise ValueError("Q3 module_summary base_seed does not match")
        if summary.get("nsga2_seeds") != derived_seeds:
            raise ValueError("Q3 derived NSGA-II seeds do not match their records")
    return len(records)


def write_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write actual-timing JSON records below the isolated toy result directory."""

    payloads = build_artifacts(seed=seed) if artifacts is None else dict(artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for filename, payload in payloads.items():
        path = output_dir / filename
        path.write_text(_json_text(payload), encoding="utf-8")
        written[filename] = path
    return written


def check_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    expected_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Re-run and compare deterministic fields, reporting missing or stale files."""

    expected = (
        build_artifacts(seed=seed)
        if expected_artifacts is None
        else dict(expected_artifacts)
    )
    issues: list[str] = []
    valid_record_count = 0
    for filename, expected_payload in expected.items():
        path = output_dir / filename
        if not path.is_file():
            issues.append(f"missing artifact: {filename}")
            continue
        try:
            actual_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"invalid artifact {filename}: {type(error).__name__}")
            continue
        try:
            valid_record_count += _validate_artifact_payload(
                filename,
                actual_payload,
                seed=seed,
            )
        except (TypeError, ValueError) as error:
            issues.append(f"invalid artifact {filename}: {error}")
            continue
        normalized_actual = normalize_for_check(actual_payload)
        normalized_expected = normalize_for_check(expected_payload)
        differences = _difference_descriptions(
            normalized_actual,
            normalized_expected,
        )
        if differences:
            preview = ", ".join(differences[:8])
            if len(differences) > 8:
                preview += f", ... (+{len(differences) - 8} more)"
            issues.append(f"stale artifact: {filename} (differing fields: {preview})")
    unexpected = sorted(
        path.name for path in output_dir.glob("*.json") if path.name not in expected
    )
    issues.extend(f"unexpected artifact: {filename}" for filename in unexpected)
    if valid_record_count != _EXPECTED_TOTAL_RECORDS:
        issues.append(
            "invalid artifact distribution: "
            f"expected {_EXPECTED_TOTAL_RECORDS} records, found {valid_record_count}"
        )
    return not issues, tuple(issues)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-run and verify committed deterministic fields without rewriting files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    arguments = _parser().parse_args(argv)
    if arguments.check:
        ok, issues = check_artifacts(
            output_dir=arguments.output_dir,
            seed=arguments.seed,
        )
        if ok:
            print("toy artifacts are current")
            return 0
        for issue in issues:
            print(issue)
        return 1
    artifacts = build_artifacts(seed=arguments.seed)
    written = write_artifacts(
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        artifacts=artifacts,
    )
    if not all(path.is_file() for path in written.values()):
        return 1
    elapsed = sum(
        record["runtime_s"]
        for payload in artifacts.values()
        for record in payload["records"]
    )
    print(f"wrote {len(written)} toy artifacts (reported solver runtime {elapsed:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
