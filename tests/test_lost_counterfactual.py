from __future__ import annotations

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
