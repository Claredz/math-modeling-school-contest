from smoke_defense.constants import load_problem_constants


def test_problem_constants_match_statement():
    constants = load_problem_constants()

    assert constants.ship.speed_mps == 7.71
    assert constants.ship.effective_radius_m == 80.0
    assert constants.uav.speed_mps == 28.0
    assert constants.uav.operation_radius_m == 12000.0
    assert constants.uav.max_payload_count == 3
    assert constants.missile.nominal_speed_mps == 320.0
    assert constants.missile.detection_range_m == 8000.0
    assert constants.missile.field_of_view_half_angle_deg == 15.0
    assert constants.countermeasure.release_response_min_s == 2.0
    assert constants.countermeasure.minimum_release_interval_s == 1.0
    assert constants.countermeasure.detonation_delay_s == 3.5
    assert constants.smoke.maximum_radius_m == 120.0
    assert constants.smoke.hold_duration_s == 18.0
    assert constants.smoke.decay_duration_s == 5.0
