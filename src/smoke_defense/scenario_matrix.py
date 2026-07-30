"""Generate the approved Q1-Q3 geometry and guidance parameter matrices."""

from __future__ import annotations

from itertools import product
from math import cos, radians, sin
from pathlib import Path

import yaml

from smoke_defense.scenario import Scenario

DEFAULT_GUIDANCE_SWEEP_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "sweeps" / "guidance.yaml"
)

DISTANCES_M = (8000.0, 10000.0, 12000.0, 15000.0)
DIRECTIONS = {
    "front": (1.0, 0.0),
    "rear": (-1.0, 0.0),
    "side": (0.0, 1.0),
    "oblique": (cos(radians(135.0)), sin(radians(135.0))),
}
ASSUMPTION_IDS = (
    "A-001",
    "A-002",
    "A-003",
    "A-019",
    "A-020",
    "A-021",
    "A-022",
)


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _base_scene(
    *,
    scenario_id: str,
    position_body_m: tuple[float, float],
    model_layer: str,
    guidance_model: str,
    heading_response_rate_per_s: float | None = None,
    max_turn_rate_deg_s: float | None = None,
) -> Scenario:
    missile = {
        "id": "M1",
        "appearance_time_s": 0.0,
        "initial_position_at_appearance_body_m": position_body_m,
        "guidance_model": guidance_model,
        "optical_axis_model": "velocity_aligned",
    }
    if heading_response_rate_per_s is not None:
        missile["heading_response_rate_per_s"] = heading_response_rate_per_s
    if max_turn_rate_deg_s is not None:
        missile["max_turn_rate_deg_s"] = max_turn_rate_deg_s
    return Scenario.model_validate(
        {
            "schema_version": "1.0",
            "constants_version": "b-problem-v2",
            "scenario_id": scenario_id,
            "time_origin": "decision_start",
            "model_layer": model_layer,
            "assumption_ids": ASSUMPTION_IDS,
            "ship": {
                "initial_position_world_m": (0.0, 0.0),
                "heading_deg": 0.0,
            },
            "uavs": [{"id": "U1", "available_time_s": 0.0}],
            "missiles": [missile],
            "constraints": {"safe_distance_m": 100.0},
        }
    )


def _position(direction: tuple[float, float], distance_m: float) -> tuple[float, float]:
    return direction[0] * distance_m, direction[1] * distance_m


def generate_q1_q3_matrix(
    sweep_path: str | Path = DEFAULT_GUIDANCE_SWEEP_PATH,
) -> tuple[Scenario, ...]:
    """Return 4 directions x 4 distances x 3 response rates x 3 turn rates."""

    with Path(sweep_path).open(encoding="utf-8") as stream:
        sweep = yaml.safe_load(stream)
    response_rates = sweep["heading_response_rate_per_s"]
    turn_rates = sweep["max_turn_rate_deg_s"]
    scenes = []
    for (direction_name, direction), distance_m, response_rate, turn_rate in product(
        DIRECTIONS.items(),
        DISTANCES_M,
        response_rates,
        turn_rates,
    ):
        scenes.append(
            _base_scene(
                scenario_id=(
                    f"q1_q3_{direction_name}_d{int(distance_m)}"
                    f"_k{_token(response_rate)}_w{_token(turn_rate)}"
                ),
                position_body_m=_position(direction, distance_m),
                model_layer="formal",
                guidance_model="inertial_pure_pursuit",
                heading_response_rate_per_s=response_rate,
                max_turn_rate_deg_s=turn_rate,
            )
        )
    return tuple(scenes)


def generate_instantaneous_ablation_matrix() -> tuple[Scenario, ...]:
    """Return one instantaneous-pursuit reference for each geometry."""

    scenes = []
    for (direction_name, direction), distance_m in product(
        DIRECTIONS.items(),
        DISTANCES_M,
    ):
        scenes.append(
            _base_scene(
                scenario_id=f"ablation_{direction_name}_d{int(distance_m)}",
                position_body_m=_position(direction, distance_m),
                model_layer="ablation",
                guidance_model="instantaneous_pure_pursuit",
            )
        )
    return tuple(scenes)
