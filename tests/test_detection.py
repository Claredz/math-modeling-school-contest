from dataclasses import dataclass
from math import radians

import numpy as np
import pytest

from smoke_defense.detection import build_detection_set
from smoke_defense.events import ClosedInterval, EventKind


@dataclass(frozen=True)
class FunctionalTrajectory:
    start_time_s: float
    end_time_s: float
    hit_time_s: float | None
    position_function: object
    heading_function: object

    def position(self, time_s: float) -> np.ndarray:
        return np.asarray(self.position_function(time_s), dtype=float)

    def heading(self, time_s: float) -> float:
        return float(self.heading_function(time_s))


def origin(_time_s: float) -> np.ndarray:
    return np.zeros(2)


def test_missile_does_not_exist_before_appearance():
    trajectory = FunctionalTrajectory(
        start_time_s=2.0,
        end_time_s=5.0,
        hit_time_s=None,
        position_function=lambda time_s: np.array([10.0 - time_s, 0.0]),
        heading_function=lambda _time_s: np.pi,
    )
    detection = build_detection_set(
        trajectory,
        origin,
        detection_range_m=20.0,
        field_of_view_half_angle_rad=radians(15.0),
    )

    assert detection.contains(2.0)
    assert not detection.contains(2.0 - 1e-9)
    assert detection.components[0].start_s == pytest.approx(2.0)


def test_distance_entry_is_found_as_continuous_event():
    trajectory = FunctionalTrajectory(
        start_time_s=0.0,
        end_time_s=5.0,
        hit_time_s=None,
        position_function=lambda time_s: np.array([10.0 - time_s, 0.0]),
        heading_function=lambda _time_s: np.pi,
    )
    detection = build_detection_set(
        trajectory,
        origin,
        detection_range_m=8.0,
        field_of_view_half_angle_rad=radians(15.0),
        event_scan_step_s=0.7,
    )

    assert detection.components == (ClosedInterval(2.0, 5.0),)
    entry = [
        event
        for event in detection.source_events
        if event.kind is EventKind.DISTANCE_ENTRY
    ]
    assert entry[0].time_s == pytest.approx(2.0, abs=1e-10)


def test_fov_entry_and_exit_split_detection_components():
    trajectory = FunctionalTrajectory(
        start_time_s=0.0,
        end_time_s=6.0,
        hit_time_s=None,
        position_function=lambda _time_s: np.array([1.0, 0.0]),
        heading_function=lambda time_s: np.pi
        + radians(20.0) * np.sin(np.pi * time_s / 2.0),
    )
    detection = build_detection_set(
        trajectory,
        origin,
        detection_range_m=8.0,
        field_of_view_half_angle_rad=radians(15.0),
        event_scan_step_s=0.05,
    )

    assert len(detection.components) >= 2
    kinds = {event.kind for event in detection.source_events}
    assert EventKind.FOV_ENTRY in kinds
    assert EventKind.FOV_EXIT in kinds


def test_hit_time_is_included_as_closed_detection_endpoint():
    trajectory = FunctionalTrajectory(
        start_time_s=0.0,
        end_time_s=4.25,
        hit_time_s=4.25,
        position_function=lambda time_s: np.array([10.0 - time_s, 0.0]),
        heading_function=lambda _time_s: np.pi,
    )
    detection = build_detection_set(
        trajectory,
        origin,
        detection_range_m=20.0,
        field_of_view_half_angle_rad=radians(15.0),
    )

    assert detection.components == (ClosedInterval(0.0, 4.25),)
    assert detection.contains(4.25)
    assert detection.source_events[-1].kind is EventKind.HIT


def test_initial_centre_hit_records_events_without_evaluating_line_of_sight():
    trajectory = FunctionalTrajectory(
        start_time_s=0.0,
        end_time_s=0.0,
        hit_time_s=0.0,
        position_function=lambda _time_s: np.zeros(2),
        heading_function=lambda _time_s: 0.0,
    )

    detection = build_detection_set(
        trajectory,
        origin,
        detection_range_m=20.0,
        field_of_view_half_angle_rad=radians(15.0),
    )

    assert detection.components == ()
    assert [event.time_s for event in detection.source_events] == pytest.approx(
        [0.0, 0.0]
    )
    assert [event.kind for event in detection.source_events] == [
        EventKind.APPEARANCE,
        EventKind.HIT,
    ]


def test_detection_set_can_have_two_disconnected_closed_components():
    trajectory = FunctionalTrajectory(
        start_time_s=0.0,
        end_time_s=5.0,
        hit_time_s=None,
        position_function=lambda _time_s: np.array([1.0, 0.0]),
        heading_function=lambda time_s: np.pi
        + radians(25.0) * np.sin(np.pi * time_s / 2.0),
    )
    detection = build_detection_set(
        trajectory,
        origin,
        detection_range_m=8.0,
        field_of_view_half_angle_rad=radians(15.0),
        event_scan_step_s=0.02,
    )

    assert len(detection.components) > 1
    assert all(component.start_s <= component.end_s for component in detection.components)
