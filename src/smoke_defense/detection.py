"""Distance-and-field-of-view detection sets for continuous trajectories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from smoke_defense.angles import wrap_to_pi
from smoke_defense.events import (
    ClosedInterval,
    EventKind,
    TrajectoryEvent,
    find_boundary_events,
    intervals_where_nonnegative,
)

PositionFunction = Callable[[float], np.ndarray]


class HeadingTrajectory(Protocol):
    start_time_s: float
    end_time_s: float
    hit_time_s: float | None

    def position(self, time_s: float) -> np.ndarray: ...

    def heading(self, time_s: float) -> float: ...


@dataclass(frozen=True)
class DetectionSet:
    components: tuple[ClosedInterval, ...]
    source_events: tuple[TrajectoryEvent, ...]

    @property
    def duration_s(self) -> float:
        return sum(component.duration_s for component in self.components)

    def contains(self, time_s: float) -> bool:
        return any(component.contains(time_s) for component in self.components)


def line_of_sight_angle(
    missile_position_m: np.ndarray,
    ship_position_m: np.ndarray,
) -> float:
    relative = np.asarray(ship_position_m) - np.asarray(missile_position_m)
    if np.linalg.norm(relative) == 0:
        raise ValueError("line of sight is undefined at zero relative distance")
    return float(np.arctan2(relative[1], relative[0]))


def field_of_view_error_rad(
    trajectory: HeadingTrajectory,
    ship_position: PositionFunction,
    time_s: float,
) -> float:
    line_of_sight = line_of_sight_angle(
        trajectory.position(time_s),
        ship_position(time_s),
    )
    return wrap_to_pi(line_of_sight - trajectory.heading(time_s))


def build_detection_set(
    trajectory: HeadingTrajectory,
    ship_position: PositionFunction,
    *,
    detection_range_m: float,
    field_of_view_half_angle_rad: float,
    event_scan_step_s: float = 0.05,
) -> DetectionSet:
    if detection_range_m <= 0:
        raise ValueError("detection range must be positive")
    if not 0 < field_of_view_half_angle_rad <= np.pi:
        raise ValueError("field-of-view half angle must lie in (0, pi]")

    start_s = float(trajectory.start_time_s)
    end_s = float(trajectory.end_time_s)

    def distance_margin(time_s: float) -> float:
        separation = np.linalg.norm(
            trajectory.position(time_s) - ship_position(time_s)
        )
        return float(detection_range_m - separation)

    def fov_margin(time_s: float) -> float:
        error = field_of_view_error_rad(trajectory, ship_position, time_s)
        return float(field_of_view_half_angle_rad - abs(error))

    def detection_margin(time_s: float) -> float:
        if trajectory.hit_time_s is not None and time_s == trajectory.hit_time_s:
            # The line of sight remains defined at contact with the positive hit radius.
            return min(distance_margin(time_s), fov_margin(time_s))
        return min(distance_margin(time_s), fov_margin(time_s))

    components = intervals_where_nonnegative(
        detection_margin,
        start_s=start_s,
        end_s=end_s,
        max_step_s=event_scan_step_s,
    )
    events: list[TrajectoryEvent] = [
        TrajectoryEvent(start_s, EventKind.APPEARANCE)
    ]
    events.extend(
        find_boundary_events(
            distance_margin,
            start_s=start_s,
            end_s=end_s,
            max_step_s=event_scan_step_s,
            entry_kind=EventKind.DISTANCE_ENTRY,
            exit_kind=EventKind.DISTANCE_EXIT,
        )
    )
    events.extend(
        find_boundary_events(
            fov_margin,
            start_s=start_s,
            end_s=end_s,
            max_step_s=event_scan_step_s,
            entry_kind=EventKind.FOV_ENTRY,
            exit_kind=EventKind.FOV_EXIT,
        )
    )
    for component in components:
        if component.start_s > start_s:
            events.append(
                TrajectoryEvent(component.start_s, EventKind.DETECTION_ENTRY)
            )
        if component.end_s < end_s:
            events.append(
                TrajectoryEvent(component.end_s, EventKind.DETECTION_EXIT)
            )
    if trajectory.hit_time_s is not None:
        events.append(TrajectoryEvent(trajectory.hit_time_s, EventKind.HIT))
    events.sort(key=lambda event: (event.time_s, event.kind.value))
    return DetectionSet(components=components, source_events=tuple(events))
