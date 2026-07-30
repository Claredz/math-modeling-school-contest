from copy import deepcopy

import pytest
from pydantic import ValidationError

from smoke_defense.scenario import Scenario, load_scenario, scenario_hash


def valid_scenario_dict() -> dict:
    return {
        "schema_version": "1.0",
        "constants_version": "b-problem-v2",
        "scenario_id": "q1_front_d10000_k1_w10",
        "time_origin": "decision_start",
        "model_layer": "formal",
        "assumption_ids": [
            "A-001",
            "A-002",
            "A-003",
            "A-019",
            "A-020",
            "A-021",
            "A-022",
        ],
        "ship": {
            "initial_position_world_m": [0.0, 0.0],
            "heading_deg": 0.0,
        },
        "uavs": [{"id": "U1", "available_time_s": 0.0}],
        "missiles": [
            {
                "id": "M1",
                "appearance_time_s": 0.0,
                "initial_position_at_appearance_body_m": [10000.0, 0.0],
                "guidance_model": "inertial_pure_pursuit",
                "heading_response_rate_per_s": 1.0,
                "max_turn_rate_deg_s": 10.0,
                "optical_axis_model": "velocity_aligned",
            }
        ],
        "constraints": {"safe_distance_m": 100.0},
    }


@pytest.fixture
def valid_dict() -> dict:
    return valid_scenario_dict()


def test_formal_scene_uses_inertial_pursuit(valid_dict):
    scene = Scenario.model_validate(valid_dict)
    missile = scene.missiles[0]

    assert missile.guidance_model == "inertial_pure_pursuit"
    assert missile.heading_response_rate_per_s == pytest.approx(1.0)
    assert missile.max_turn_rate_deg_s == pytest.approx(10.0)


def test_formal_scene_rejects_instantaneous_pursuit(valid_dict):
    valid_dict["missiles"][0]["guidance_model"] = "instantaneous_pure_pursuit"
    valid_dict["missiles"][0].pop("heading_response_rate_per_s")
    valid_dict["missiles"][0].pop("max_turn_rate_deg_s")

    with pytest.raises(ValidationError):
        Scenario.model_validate(valid_dict)


def test_ablation_scene_allows_only_instantaneous_reference(valid_dict):
    valid_dict["model_layer"] = "ablation"
    valid_dict["missiles"][0]["guidance_model"] = "instantaneous_pure_pursuit"
    valid_dict["missiles"][0].pop("heading_response_rate_per_s")
    valid_dict["missiles"][0].pop("max_turn_rate_deg_s")

    scene = Scenario.model_validate(valid_dict)

    assert scene.model_layer == "ablation"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_position_world_m", [0.0, 0.0]),
        ("initial_heading_deg", 0.0),
        ("launch_offset_body_m", [0.0, 0.0]),
    ],
)
def test_uav_free_initial_state_is_rejected(valid_dict, field, value):
    valid_dict["uavs"][0][field] = value

    with pytest.raises(ValidationError):
        Scenario.model_validate(valid_dict)


def test_missile_requires_exactly_one_initial_position(valid_dict):
    missing = deepcopy(valid_dict)
    missing["missiles"][0].pop("initial_position_at_appearance_body_m")
    duplicate = deepcopy(valid_dict)
    duplicate["missiles"][0]["initial_position_world_m"] = [10000.0, 0.0]

    with pytest.raises(ValidationError):
        Scenario.model_validate(missing)
    with pytest.raises(ValidationError):
        Scenario.model_validate(duplicate)


def test_speed_override_requires_source(valid_dict):
    valid_dict["missiles"][0]["speed_override_mps"] = 300.0

    with pytest.raises(ValidationError):
        Scenario.model_validate(valid_dict)


def test_scenario_hash_is_order_stable_and_parameter_sensitive(valid_dict):
    first = Scenario.model_validate(valid_dict)
    reordered = Scenario.model_validate(dict(reversed(list(valid_dict.items()))))
    changed_dict = deepcopy(valid_dict)
    changed_dict["missiles"][0]["max_turn_rate_deg_s"] = 20.0
    changed = Scenario.model_validate(changed_dict)

    assert scenario_hash(first) == scenario_hash(reordered)
    assert scenario_hash(first) != scenario_hash(changed)


def test_example_scenario_loads():
    scene = load_scenario(
        "configs/scenarios/examples/q1_front_d10000_k1_w10.yaml"
    )

    assert scene.scenario_id == "q1_front_d10000_k1_w10"
