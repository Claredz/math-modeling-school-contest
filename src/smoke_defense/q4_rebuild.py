"""Q4 causal rolling allocation over certified task packages."""

from __future__ import annotations

from dataclasses import dataclass


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


def schedule_causal_rolling(
    tasks: tuple[ThreatTask, ...],
    *,
    resource_capacity: int,
    horizon_end_s: float,
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
        feasible = [item for item in pending if item.resource_cost <= remaining]
        if feasible:
            selected = max(
                feasible,
                key=lambda item: (item.value / item.resource_cost, item.value, item.task_id),
            )
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
    for task in (*pending, *ordered[index:]):
        if task.certified:
            unresolved.append(task.task_id)
    total_value = sum(item.value for item in decisions)
    certified_value = sum(
        item.value
        for item in decisions
        if next(task for task in ordered if task.task_id == item.task_id).certified
    )
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
    )


def build_q4_tasks() -> tuple[ThreatTask, ...]:
    """Small deterministic package set for the Q4 paper-facing benchmark."""

    return (
        ThreatTask("threat_front", 0.0, 17.2, 1, 17.2),
        ThreatTask("threat_rear", 4.0, 15.8, 1, 15.8),
        ThreatTask("threat_side", 8.0, 13.4, 2, 13.4),
        ThreatTask("threat_oblique", 12.0, 11.1, 1, 11.1),
    )
