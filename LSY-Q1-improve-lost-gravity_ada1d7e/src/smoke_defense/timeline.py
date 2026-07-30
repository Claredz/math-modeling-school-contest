"""Hybrid command, release, burst, and trajectory event timeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from smoke_defense.events import EventKind, TrajectoryEvent


def earliest_release_time(
    command_time_s: float,
    minimum_response_s: float = 2.0,
) -> float:
    if minimum_response_s < 0:
        raise ValueError("minimum response time cannot be negative")
    return command_time_s + minimum_response_s


@dataclass(frozen=True)
class BombEvents:
    command_time_s: float
    release_time_s: float
    burst_time_s: float
    minimum_response_s: float = 2.0
    detonation_delay_s: float = 3.5

    def __post_init__(self) -> None:
        minimum_release = earliest_release_time(
            self.command_time_s,
            self.minimum_response_s,
        )
        if self.release_time_s < minimum_release:
            raise ValueError("release violates the minimum command response time")
        actual_delay = self.burst_time_s - self.release_time_s
        if not isclose(
            actual_delay,
            self.detonation_delay_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("burst violates the nominal detonation delay")

    @property
    def timeline_events(self) -> tuple[TrajectoryEvent, ...]:
        return (
            TrajectoryEvent(self.command_time_s, EventKind.COMMAND),
            TrajectoryEvent(self.release_time_s, EventKind.RELEASE),
            TrajectoryEvent(self.burst_time_s, EventKind.BURST),
        )


@dataclass(frozen=True)
class HybridTimeline:
    events: tuple[TrajectoryEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "events",
            tuple(sorted(self.events, key=lambda event: event.time_s)),
        )

    @property
    def times_s(self) -> tuple[float, ...]:
        return tuple(event.time_s for event in self.events)

    @classmethod
    def combine(
        cls,
        *event_groups: tuple[TrajectoryEvent, ...],
    ) -> HybridTimeline:
        return cls(tuple(event for group in event_groups for event in group))
