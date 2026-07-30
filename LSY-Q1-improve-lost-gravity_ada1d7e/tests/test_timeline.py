import pytest

from smoke_defense.events import EventKind, TrajectoryEvent
from smoke_defense.timeline import (
    BombEvents,
    HybridTimeline,
    earliest_release_time,
)


def test_release_later_than_minimum_response_is_allowed():
    events = BombEvents(
        command_time_s=4.0,
        release_time_s=7.3,
        burst_time_s=10.8,
    )

    assert earliest_release_time(4.0) == pytest.approx(6.0)
    assert events.release_time_s == pytest.approx(7.3)


def test_release_before_minimum_response_is_rejected():
    with pytest.raises(ValueError, match="response"):
        BombEvents(
            command_time_s=4.0,
            release_time_s=5.9,
            burst_time_s=9.4,
        )


def test_nominal_detonation_delay_must_equal_inertial_flight_time():
    with pytest.raises(ValueError, match="detonation"):
        BombEvents(
            command_time_s=4.0,
            release_time_s=7.3,
            burst_time_s=10.7,
        )


def test_hybrid_timeline_sorts_physical_event_times():
    timeline = HybridTimeline(
        (
            TrajectoryEvent(5.5, EventKind.BURST),
            TrajectoryEvent(0.0, EventKind.APPEARANCE),
            TrajectoryEvent(2.0, EventKind.RELEASE),
            TrajectoryEvent(0.0, EventKind.COMMAND),
        )
    )

    assert timeline.times_s == (0.0, 0.0, 2.0, 5.5)
