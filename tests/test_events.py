from math import pi

import pytest

from smoke_defense.events import (
    ClosedInterval,
    EventKind,
    find_boundary_events,
    intervals_where_nonnegative,
)


def test_boundary_time_is_root_solved_not_sample_index_time():
    events = find_boundary_events(
        lambda time_s: time_s - pi,
        start_s=0.0,
        end_s=5.0,
        max_step_s=1.0,
        entry_kind=EventKind.DISTANCE_ENTRY,
        exit_kind=EventKind.DISTANCE_EXIT,
    )

    assert len(events) == 1
    assert events[0].time_s == pytest.approx(pi, abs=1e-10)
    assert events[0].time_s != pytest.approx(3.0)
    assert events[0].kind is EventKind.DISTANCE_ENTRY


def test_nonnegative_intervals_are_closed_and_can_be_disconnected():
    intervals = intervals_where_nonnegative(
        lambda time_s: ((time_s - 1.0) * (time_s - 2.0))
        * ((time_s - 3.0) * (time_s - 4.0)),
        start_s=0.0,
        end_s=5.0,
        max_step_s=0.1,
    )

    assert intervals == (
        ClosedInterval(0.0, 1.0),
        ClosedInterval(2.0, 3.0),
        ClosedInterval(4.0, 5.0),
    )


def test_closed_interval_rejects_reversed_endpoints():
    with pytest.raises(ValueError, match="end"):
        ClosedInterval(start_s=2.0, end_s=1.0)
