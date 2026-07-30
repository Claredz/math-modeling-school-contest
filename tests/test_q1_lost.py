import numpy as np

from smoke_defense.q1_lost import solve_q1_lost_coupled
from smoke_defense.scenario_matrix import generate_q1_q3_matrix


def _side_12000_k1_w5():
    return next(
        scenario
        for scenario in generate_q1_q3_matrix()
        if scenario.scenario_id == "q1_q3_side_d12000_k1_w5"
    )


def test_improved_q1_has_a_reachable_permanent_loss_solution():
    result = solve_q1_lost_coupled(
        _side_12000_k1_w5(),
        initial_heading_error_deg=10.0,
        lost_turn_decay_time_s=5.0,
        burst_times_s=tuple(np.arange(14.0, 19.01, 0.5)),
        time_step_s=0.05,
    )

    assert result.feasible
    assert result.unique_optimum_on_search_grid
    best = result.best_candidate
    assert best is not None
    assert best.trajectory.hit_time_s is None
    assert best.trajectory.escaped_without_reacquisition
    assert best.release_time_s >= 2.0
    assert best.burst_time_s - best.release_time_s == 3.5
    np.testing.assert_allclose(best.burst_center_m, best.path.ship.position(best.burst_time_s))
