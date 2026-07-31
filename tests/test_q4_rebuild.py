import json
from pathlib import Path

from smoke_defense.q4_rebuild import (
    ThreatTask,
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


def test_q4_uncertified_high_value_task_is_never_scheduled():
    tasks = (
        ThreatTask("uncertified_high", 0.0, 100.0, 1, 0.0, certified=False),
        ThreatTask("certified_later", 1.0, 10.0, 1, 10.0, certified=True),
    )

    for scheduler in (schedule_causal_rolling, schedule_causal_greedy):
        result = scheduler(tasks, resource_capacity=1, horizon_end_s=10.0)
        assert [item.task_id for item in result.decisions] == ["certified_later"]
        assert result.total_value == result.certified_value == 10.0


def test_q4_uncertified_task_is_retained_as_unresolved_and_does_not_block():
    tasks = (
        ThreatTask("uncertified_high", 0.0, 100.0, 1, 0.0, certified=False),
        ThreatTask("certified_later", 1.0, 10.0, 1, 10.0, certified=True),
    )

    result = schedule_causal_rolling(tasks, resource_capacity=1, horizon_end_s=10.0)

    assert "uncertified_high" in result.unresolved_task_ids
    assert "uncertified_high" not in {item.task_id for item in result.decisions}
    assert result.status == "unresolved"


def test_q4_hindsight_filters_uncertified_tasks_before_optimization():
    tasks = (
        ThreatTask("uncertified_high", 0.0, 100.0, 1, 0.0, certified=False),
        ThreatTask("certified_later", 1.0, 10.0, 1, 10.0, certified=True),
    )

    result = schedule_offline_hindsight(tasks, resource_capacity=1, horizon_end_s=10.0)

    assert [item.task_id for item in result.decisions] == ["certified_later"]
    assert "uncertified_high" in result.unresolved_task_ids
    assert result.total_value == result.certified_value == 10.0


def test_q4_all_certified_tasks_preserve_existing_schedule_behavior():
    tasks = build_q4_tasks()

    rolling = schedule_causal_rolling(tasks, resource_capacity=2, horizon_end_s=30.0)
    greedy = schedule_causal_greedy(tasks, resource_capacity=2, horizon_end_s=30.0)
    hindsight = schedule_offline_hindsight(tasks, resource_capacity=2, horizon_end_s=30.0)

    assert [item.task_id for item in rolling.decisions] == ["threat_front", "threat_rear"]
    assert [item.task_id for item in greedy.decisions] == ["threat_front", "threat_rear"]
    assert [item.task_id for item in hindsight.decisions] == ["threat_front", "threat_rear"]
    assert rolling.total_value == rolling.certified_value == 33.0
    assert rolling.unresolved_task_ids == ("threat_side", "threat_oblique")


def test_q4_generated_artifact_keeps_unresolved_and_certified_value_semantics():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "results/q4_rebuild/q4_results.json").read_text(encoding="utf-8")
    )

    for row in payload["resource_cases"]:
        assert row["total_value"] == row["certified_value"]
        assert set(row["unresolved_task_ids"]).isdisjoint(
            {item["task_id"] for item in row["decisions"]}
        )
