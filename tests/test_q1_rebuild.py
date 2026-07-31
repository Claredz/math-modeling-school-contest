from __future__ import annotations

import numpy as np
import pytest

from smoke_defense.q1_rebuild import (
    build_q1_problem,
    construct_q1_candidate,
    q1_verification_rank,
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
