from __future__ import annotations

from scripts.run_q1_rebuild import build_formal_counterfactual_payload
from smoke_defense.lost_counterfactual import (
    LostCounterfactualParameters,
    simulate_lost_counterfactual,
)
from smoke_defense.q1_rebuild import build_q1_problem, construct_q1_candidate
from smoke_defense.scenario_matrix import generate_q1_rebuild_matrix


def test_lost_counterfactual_is_explicitly_nonformal():
    scenario = generate_q1_rebuild_matrix()[0]
    problem = build_q1_problem(scenario)
    candidate = construct_q1_candidate(problem, burst_time_s=12.0, center_time_s=12.0)

    result = simulate_lost_counterfactual(
        problem,
        candidate,
        LostCounterfactualParameters(tau_t_s=0.5, tau_l_s=5.0, t_r_s=1.0),
    )

    assert result.label == "experimental_counterfactual"
    assert result.formal_baseline is False
    assert isinstance(result.lost, bool)
    assert isinstance(result.reacquired, bool)
    assert result.minimum_separation_m > 0.0
    assert result.parameters == {"tau_T_s": 0.5, "tau_L_s": 5.0, "T_R_s": 1.0}


def test_counterfactual_reuses_formal_candidate_decision_and_candidate_id():
    scenario = generate_q1_rebuild_matrix()[0]
    problem = build_q1_problem(scenario)
    formal_candidate = construct_q1_candidate(problem, burst_time_s=12.0, center_time_s=12.0)
    formal_row = {
        "scenario_id": scenario.scenario_id,
        "candidate_id": "formal-q1-rebuild-front",
        "burst_time_s": formal_candidate.burst_time_s,
        "center_time_s": formal_candidate.center_time_s,
    }

    payload = build_formal_counterfactual_payload(problem, formal_row)

    assert payload["candidate_id"] == formal_row["candidate_id"]
    assert payload["formal_baseline"] is False
    assert payload["fixed_decision"] is True
    assert payload["changed_model"] is True
    assert payload["decision"]["command_time_s"] == formal_candidate.command_time_s
    assert payload["decision"]["drop_time_s"] == formal_candidate.drop_time_s
    assert payload["decision"]["burst_time_s"] == formal_candidate.burst_time_s
    assert payload["decision"]["center_time_s"] == formal_candidate.center_time_s
    assert payload["decision"]["release_position_m"] == formal_candidate.drop_position_m.tolist()
    assert payload["decision"]["uav_path"]
    assert payload["formal_candidate_id"] == payload["candidate_id"]
    assert payload["candidate_id"] not in {"formal_q1_ranked_candidate", "counterfactual_only"}
