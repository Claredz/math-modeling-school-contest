"""Q4 causal rolling allocation over certified task packages."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class ThreatTask:
    task_id: str
    reveal_time_s: float
    value: float
    resource_cost: int
    certified_coverage_lower_s: float
    certified: bool = True


@dataclass(frozen=True)
class ScheduleDecision:
    time_s: float
    task_id: str
    resources_used: int
    value: float


@dataclass(frozen=True)
class Q4ScheduleCertificate:
    status: str
    decisions: tuple[ScheduleDecision, ...]
    total_value: float
    certified_value: float
    capacity: int
    causal: bool
    unresolved_task_ids: tuple[str, ...] = ()
    reason: str = ""
    strategy: str = "causal_rolling"
    hindsight_upper_bound_value: float | None = None


def schedule_causal_rolling(
    tasks: tuple[ThreatTask, ...],
    *,
    resource_capacity: int,
    horizon_end_s: float,
    selection_rule: str = "value_per_resource",
) -> Q4ScheduleCertificate:
    """Schedule only tasks revealed by the current rolling time."""

    if resource_capacity <= 0:
        raise ValueError("resource capacity must be positive")
    ordered = tuple(sorted(tasks, key=lambda item: (item.reveal_time_s, item.task_id)))
    decisions: list[ScheduleDecision] = []
    unresolved: list[str] = []
    current_time = 0.0
    remaining = resource_capacity
    pending: list[ThreatTask] = []
    index = 0
    while current_time <= horizon_end_s and (index < len(ordered) or pending):
        while index < len(ordered) and ordered[index].reveal_time_s <= current_time + 1e-9:
            pending.append(ordered[index])
            index += 1
        feasible = [
            item
            for item in pending
            if item.certified and item.resource_cost <= remaining
        ]
        if feasible:
            if selection_rule == "value":
                def key(item: ThreatTask) -> tuple[float, int, str]:
                    return item.value, -item.resource_cost, item.task_id
            elif selection_rule == "value_per_resource":
                def key(item: ThreatTask) -> tuple[float, float, str]:
                    return item.value / item.resource_cost, item.value, item.task_id
            else:
                raise ValueError("unknown Q4 selection rule")
            selected = max(feasible, key=key)
            pending.remove(selected)
            remaining -= selected.resource_cost
            decisions.append(
                ScheduleDecision(
                    current_time,
                    selected.task_id,
                    selected.resource_cost,
                    selected.value,
                )
            )
            if remaining == 0:
                current_time = horizon_end_s + 1.0
            continue
        if index < len(ordered):
            next_time = ordered[index].reveal_time_s
            if next_time > horizon_end_s:
                break
            current_time = max(current_time + 1e-6, next_time)
        else:
            break
    unresolved.extend(task.task_id for task in (*pending, *ordered[index:]))
    total_value = sum(item.value for item in decisions)
    # The scheduler can only commit certified packages, so both fields now have
    # the same value.  Keeping both names preserves the artifact contract while
    # making it impossible for an uncertified package to enter total_value.
    certified_value = total_value
    status = "certified_feasible" if not unresolved else "unresolved"
    return Q4ScheduleCertificate(
        status=status,
        decisions=tuple(decisions),
        total_value=total_value,
        certified_value=certified_value,
        capacity=resource_capacity,
        causal=all(
            decision.time_s >= next(
                task.reveal_time_s for task in ordered if task.task_id == decision.task_id
            )
            for decision in decisions
        ),
        unresolved_task_ids=tuple(unresolved),
        reason=(
            "all selected packages were revealed and certified"
            if not unresolved
            else "pending tasks retained as unresolved"
        ),
        strategy=(
            "causal_greedy"
            if selection_rule == "value"
            else "causal_rolling_value_per_resource"
        ),
    )


def schedule_causal_greedy(
    tasks: tuple[ThreatTask, ...],
    *,
    resource_capacity: int,
    horizon_end_s: float,
) -> Q4ScheduleCertificate:
    """Mandatory causal greedy baseline using revealed task value only."""

    return schedule_causal_rolling(
        tasks,
        resource_capacity=resource_capacity,
        horizon_end_s=horizon_end_s,
        selection_rule="value",
    )


def schedule_offline_hindsight(
    tasks: tuple[ThreatTask, ...],
    *,
    resource_capacity: int,
    horizon_end_s: float,
) -> Q4ScheduleCertificate:
    """Small exact subset oracle used only as an offline upper bound."""

    if resource_capacity <= 0:
        raise ValueError("resource capacity must be positive")
    candidates = tuple(
        task
        for task in tasks
        if task.reveal_time_s <= horizon_end_s and task.certified
    )
    best_subset: tuple[ThreatTask, ...] = ()
    best_value = 0.0
    for count in range(len(candidates) + 1):
        for subset in combinations(candidates, count):
            if sum(item.resource_cost for item in subset) > resource_capacity:
                continue
            value = sum(item.value for item in subset)
            if value > best_value:
                best_value = value
                best_subset = subset
    decisions = tuple(
        ScheduleDecision(item.reveal_time_s, item.task_id, item.resource_cost, item.value)
        for item in sorted(best_subset, key=lambda item: item.reveal_time_s)
    )
    selected_ids = {item.task_id for item in best_subset}
    unresolved = tuple(
        task.task_id
        for task in tasks
        if task.task_id not in selected_ids
        and (not task.certified or task.reveal_time_s > horizon_end_s)
    )
    return Q4ScheduleCertificate(
        status="certified_feasible" if not unresolved else "unresolved",
        decisions=decisions,
        total_value=best_value,
        certified_value=best_value,
        capacity=resource_capacity,
        causal=False,
        unresolved_task_ids=unresolved,
        reason=(
            "offline hindsight subset oracle; not an online policy"
            if not unresolved
            else (
                "offline hindsight subset oracle; uncertified or unrevealed tasks "
                "retained as unresolved"
            )
        ),
        strategy="offline_hindsight_upper_bound",
        hindsight_upper_bound_value=best_value,
    )


def build_q4_tasks() -> tuple[ThreatTask, ...]:
    """Small deterministic package set for the Q4 paper-facing benchmark."""

    return (
        ThreatTask("threat_front", 0.0, 17.2, 1, 17.2),
        ThreatTask("threat_rear", 4.0, 15.8, 1, 15.8),
        ThreatTask("threat_side", 8.0, 13.4, 2, 13.4),
        ThreatTask("threat_oblique", 12.0, 11.1, 1, 11.1),
    )
