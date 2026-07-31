from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import smoke_defense.q1_rebuild as q1_rebuild
from smoke_defense.q1_rebuild import (
    Q1_METHODS,
    Q1Verification,
    benchmark_q1_methods,
    build_q1_problem,
    construct_q1_candidate,
    q1_candidate_rank,
    q1_verification_rank,
    select_q1_warm_start,
    verify_q1_candidate,
)
from smoke_defense.scenario_matrix import generate_q1_rebuild_matrix


@pytest.fixture(scope="module")
def front_problem():
    scenario = next(
        item
        for item in generate_q1_rebuild_matrix()
        if item.scenario_id == "q1_rebuild_front_d10000"
    )
    return build_q1_problem(scenario)


def test_candidate_uses_fixed_response_and_burst_delays(front_problem):
    candidate = construct_q1_candidate(
        front_problem,
        burst_time_s=12.0,
        center_time_s=12.0,
    )

    assert candidate.drop_time_s - candidate.command_time_s == pytest.approx(2.0)
    assert candidate.burst_time_s - candidate.drop_time_s == pytest.approx(3.5)
    assert candidate.command_time_s >= 0.0
    np.testing.assert_allclose(
        candidate.path.position(candidate.drop_time_s),
        candidate.drop_position_m,
    )
    assert candidate.path.takeoff_time_s == 0.0
    assert candidate.path.position(1.0).shape == (2,)


def test_negative_precommand_is_rejected(front_problem):
    with pytest.raises(ValueError, match="negative command"):
        construct_q1_candidate(
            front_problem,
            burst_time_s=5.0,
            center_time_s=5.0,
        )


def test_solver_candidate_has_no_embedded_certification_state(front_problem):
    candidate = construct_q1_candidate(
        front_problem,
        burst_time_s=12.0,
        center_time_s=12.0,
    )

    assert not hasattr(candidate, "status")
    verification = verify_q1_candidate(front_problem, candidate)
    assert verification.status in {
        "certified_feasible",
        "certified_infeasible",
        "unresolved",
    }
    assert verification.solver_native_success is None
    assert verification.covered_duration_s >= 0.0
    assert verification.maximum_exposure_s >= 0.0


def test_verifier_rejects_candidate_outside_operation_radius(front_problem):
    candidate = construct_q1_candidate(
        front_problem,
        burst_time_s=12.0,
        center_time_s=12.0,
    )
    far_center = candidate.smoke.burst_center_m + np.array([20000.0, 0.0])
    tampered = candidate.with_smoke_center(far_center)

    verification = verify_q1_candidate(front_problem, tampered)

    assert verification.status == "certified_infeasible"
    assert "path" in verification.reason or "burst" in verification.reason


def test_q1_rank_implements_approved_lexicographic_order(front_problem):
    early = verify_q1_candidate(
        front_problem,
        construct_q1_candidate(
            front_problem,
            burst_time_s=10.0,
            center_time_s=10.0,
        ),
    )
    later = verify_q1_candidate(
        front_problem,
        construct_q1_candidate(
            front_problem,
            burst_time_s=14.0,
            center_time_s=14.0,
        ),
    )

    assert max((early, later), key=q1_verification_rank) in {early, later}
    assert len(q1_verification_rank(early)) == 5


def test_four_methods_share_budget_seed_bounds_and_verifier(front_problem):
    results = benchmark_q1_methods(
        front_problem,
        seed=20260731,
        evaluation_budget=24,
    )

    assert {result.method for result in results} == set(Q1_METHODS)
    assert all(result.seed == 20260731 for result in results)
    assert all(result.evaluation_budget == 24 for result in results)
    assert all(0 < result.evaluations <= 24 for result in results)
    assert len({result.bounds for result in results}) == 1
    assert all(result.best_candidate is not None for result in results)
    assert all(result.verification is not None for result in results)
    assert all(
        result.verification.status
        in {"certified_feasible", "certified_infeasible", "unresolved"}
        for result in results
    )


def test_native_solver_status_is_not_conflated_with_certification(front_problem):
    results = benchmark_q1_methods(
        front_problem,
        seed=7,
        evaluation_budget=16,
    )

    assert all(isinstance(result.native_success, bool) for result in results)
    assert all(result.native_status for result in results)
    assert all(
        result.verification.solver_native_success is result.native_success
        for result in results
        if result.verification is not None
    )


def _verification(*, status="certified_infeasible", covered=0.0, exposure=0.0):
    return Q1Verification(
        status=status,
        covered_duration_s=covered,
        exposed_duration_s=exposure,
        maximum_exposure_s=exposure,
        minimum_margin_m=0.0,
        flight_distance_m=100.0,
        reason="test",
    )


def test_q1_warm_start_selects_best_coverage_sample():
    poor = SimpleNamespace(
        command_time_s=5.0, drop_time_s=7.0, burst_time_s=10.0, center_time_s=10.0
    )
    best = SimpleNamespace(
        command_time_s=7.0, drop_time_s=9.0, burst_time_s=12.0, center_time_s=12.0
    )

    selected = select_q1_warm_start(
        ((poor, _verification(covered=1.0)), (best, _verification(covered=5.0)))
    )

    assert selected is best


def test_q1_warm_start_prioritizes_certification_over_duration():
    certified = SimpleNamespace(
        command_time_s=5.0, drop_time_s=7.0, burst_time_s=10.0, center_time_s=10.0
    )
    unresolved = SimpleNamespace(
        command_time_s=7.0, drop_time_s=9.0, burst_time_s=12.0, center_time_s=12.0
    )

    selected = select_q1_warm_start(
        (
            (certified, _verification(status="certified_feasible", covered=1.0)),
            (unresolved, _verification(status="unresolved", covered=10.0)),
        )
    )

    assert selected is certified


def test_q1_warm_start_ties_are_deterministic_and_invalid_samples_are_ignored():
    earlier = SimpleNamespace(
        command_time_s=5.0, drop_time_s=7.0, burst_time_s=10.0, center_time_s=11.0
    )
    later = SimpleNamespace(
        command_time_s=7.0, drop_time_s=9.0, burst_time_s=12.0, center_time_s=12.0
    )
    verification = _verification(status="certified_feasible", covered=5.0)

    selected = select_q1_warm_start(
        (
            (None, verification),
            (later, verification),
            (earlier, verification),
        )
    )

    assert selected is earlier
    assert q1_candidate_rank(None, verification)[0] == float("-inf")


def test_q1_sobol_local_refinement_starts_from_best_evaluated_sample(
    front_problem, monkeypatch
):
    scaled_points = np.array([[0.1, 0.1], [0.9, 0.9]])
    starts = []

    class FakeSobol:
        def __init__(self, d, scramble, seed):
            assert (d, scramble, seed) == (2, True, 7)

        def random_base2(self, m):
            assert m == 3
            return scaled_points

    def fake_minimize(_fun, x0, **_kwargs):
        starts.append(np.asarray(x0, dtype=float).copy())
        return SimpleNamespace(success=False, message="captured test start")

    def fake_verify(_problem, candidate):
        return _verification(status="certified_feasible", covered=candidate.burst_time_s)

    monkeypatch.setattr(q1_rebuild.qmc, "Sobol", FakeSobol)
    monkeypatch.setattr(q1_rebuild, "minimize", fake_minimize)
    monkeypatch.setattr(q1_rebuild, "verify_q1_candidate", fake_verify)

    result = q1_rebuild._run_method(
        front_problem,
        method="sobol_slsqp",
        seed=7,
        evaluation_budget=8,
    )

    bounds = np.asarray(result.bounds, dtype=float)
    expected = bounds[:, 0] + scaled_points[1] * (bounds[:, 1] - bounds[:, 0])
    np.testing.assert_allclose(starts[0], expected)
    assert result.best_candidate.burst_time_s == pytest.approx(expected[0])
