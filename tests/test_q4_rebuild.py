from smoke_defense.q4_rebuild import (
    build_q4_tasks,
    schedule_causal_greedy,
    schedule_causal_rolling,
    schedule_offline_hindsight,
)


def test_q4_rolling_decisions_are_causal():
    result = schedule_causal_rolling(
        build_q4_tasks(), resource_capacity=2, horizon_end_s=30.0
    )
    assert result.causal
    assert result.status in {"certified_feasible", "unresolved"}
    assert result.decisions


def test_q4_rejects_nonpositive_capacity():
    try:
        schedule_causal_rolling(build_q4_tasks(), resource_capacity=0, horizon_end_s=30.0)
    except ValueError as exc:
        assert "capacity" in str(exc)
    else:
        raise AssertionError("nonpositive capacity should be rejected")


def test_q4_reports_causal_greedy_and_hindsight_upper_bound():
    tasks = build_q4_tasks()
    greedy = schedule_causal_greedy(tasks, resource_capacity=2, horizon_end_s=30.0)
    hindsight = schedule_offline_hindsight(tasks, resource_capacity=2, horizon_end_s=30.0)
    assert greedy.causal is True
    assert greedy.strategy == "causal_greedy"
    assert hindsight.causal is False
    assert hindsight.strategy == "offline_hindsight_upper_bound"
    assert hindsight.certified_value >= greedy.certified_value
