import numpy as np

from smoke_defense.dynamics import ShipMotion
from smoke_defense.lost_guidance import (
    GuidanceMode,
    LostGuidanceSpec,
    ship_fully_occluded,
    simulate_lost_coupled_missile,
)
from smoke_defense.smoke import SmokeCloud


def test_full_ship_disk_is_required_to_trigger_smoke_loss():
    smoke = SmokeCloud(burst_time_s=0.0, burst_center_m=np.zeros(2))

    assert ship_fully_occluded(np.zeros(2), smoke, 1.0, ship_radius_m=80.0)
    assert not ship_fully_occluded(
        np.array([41.0, 0.0]), smoke, 1.0, ship_radius_m=80.0
    )


def test_no_smoke_reference_reacquires_and_hits():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    result = simulate_lost_coupled_missile(
        initial_position_m=(10000.0, 0.0),
        appearance_time_s=0.0,
        ship=ship,
        smoke=None,
        spec=LostGuidanceSpec(reacquisition_confirm_s=0.0),
        final_time_s=60.0,
    )

    assert result.hit_time_s is not None
    assert not result.successful_defense
    assert GuidanceMode.TRACKED in result.modes


def test_smoke_loss_forbids_instant_live_target_steering():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    smoke = SmokeCloud(
        burst_time_s=1.0,
        burst_center_m=ship.position(1.0),
        hold_duration_s=18.0,
        decay_duration_s=5.0,
    )
    result = simulate_lost_coupled_missile(
        initial_position_m=(0.0, 1000.0),
        appearance_time_s=0.0,
        ship=ship,
        smoke=smoke,
        spec=LostGuidanceSpec(reacquisition_confirm_s=0.5),
        final_time_s=4.0,
        time_step_s=0.01,
    )

    loss_events = [event for event in result.events if event.kind == "smoke_loss"]
    assert loss_events
    loss_index = next(
        index for index, mode in enumerate(result.modes) if mode is GuidanceMode.LOST
    )
    # Turn rate remains continuous at loss and then decays; it does not jump to
    # a fresh line-of-sight command while the target is hidden.
    assert abs(result.states[loss_index, 3]) < np.deg2rad(10.0) + 1e-8
    assert abs(result.states[-1, 3]) <= abs(result.states[loss_index, 3]) + 1e-8


def test_reacquisition_requires_continuous_confirmation():
    ship = ShipMotion((0.0, 0.0), heading_rad=0.0, speed_mps=7.71)
    smoke = SmokeCloud(
        burst_time_s=0.5,
        burst_center_m=ship.position(0.5),
        hold_duration_s=0.5,
        decay_duration_s=0.1,
    )
    result = simulate_lost_coupled_missile(
        initial_position_m=(0.0, 2000.0),
        appearance_time_s=0.0,
        ship=ship,
        smoke=smoke,
        spec=LostGuidanceSpec(reacquisition_confirm_s=0.5),
        final_time_s=2.0,
        time_step_s=0.01,
    )

    loss_time = next(event.time_s for event in result.events if event.kind == "smoke_loss")
    reacquisition_time = next(
        event.time_s for event in result.events if event.kind == "reacquisition"
    )
    assert reacquisition_time - loss_time >= 0.5
