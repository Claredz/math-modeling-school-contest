from smoke_defense.q1_lost import make_custom_q1_scenario


def test_custom_scenario_preserves_judge_missile_coordinates():
    scenario = make_custom_q1_scenario(
        scenario_id="judge",
        missile_position_world_m=(3000.0, -4000.0),
        heading_response_rate_per_s=0.8,
        max_turn_rate_deg_s=7.0,
    )

    missile = scenario.missiles[0]
    assert missile.initial_position_world_m == (3000.0, -4000.0)
    assert missile.heading_response_rate_per_s == 0.8
    assert missile.max_turn_rate_deg_s == 7.0
