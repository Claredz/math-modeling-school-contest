"""Strongly typed scenario contracts and stable provenance hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from smoke_defense.constants import DEFAULT_CONSTANTS_PATH

Vector2 = tuple[float, float]
GuidanceModel = Literal[
    "inertial_pure_pursuit",
    "instantaneous_pure_pursuit",
]
ModelLayer = Literal["formal", "ablation"]


class FrozenModel(BaseModel):
    """Reject unknown scenario fields and make validated scenarios immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ShipSpec(FrozenModel):
    initial_position_world_m: Vector2
    heading_deg: float


class UavSpec(FrozenModel):
    id: str = Field(min_length=1)
    available_time_s: float = Field(ge=0)


class MissileSpec(FrozenModel):
    id: str = Field(min_length=1)
    appearance_time_s: float = Field(ge=0)
    initial_position_at_appearance_body_m: Vector2 | None = None
    initial_position_world_m: Vector2 | None = None
    speed_override_mps: float | None = Field(default=None, gt=0)
    speed_source: str | None = None
    guidance_model: GuidanceModel
    heading_response_rate_per_s: float | None = Field(default=None, gt=0)
    max_turn_rate_deg_s: float | None = Field(default=None, gt=0, le=180)
    optical_axis_model: Literal["velocity_aligned"]

    @model_validator(mode="after")
    def validate_position_and_speed_source(self) -> MissileSpec:
        positions = (
            self.initial_position_at_appearance_body_m,
            self.initial_position_world_m,
        )
        if sum(position is not None for position in positions) != 1:
            raise ValueError("exactly one missile initial-position field is required")
        if (self.speed_override_mps is None) != (self.speed_source is None):
            raise ValueError("speed_override_mps and speed_source must be supplied together")
        return self


class ConstraintSpec(FrozenModel):
    safe_distance_m: float = Field(ge=0)


class Scenario(FrozenModel):
    schema_version: Literal["1.0"]
    constants_version: str
    scenario_id: str = Field(min_length=1)
    time_origin: Literal["decision_start"]
    model_layer: ModelLayer
    assumption_ids: tuple[str, ...] = Field(min_length=1)
    ship: ShipSpec
    uavs: tuple[UavSpec, ...] = Field(min_length=1)
    missiles: tuple[MissileSpec, ...] = Field(min_length=1)
    constraints: ConstraintSpec

    @model_validator(mode="after")
    def validate_model_layer(self) -> Scenario:
        for missile in self.missiles:
            if self.model_layer == "formal":
                if missile.guidance_model != "inertial_pure_pursuit":
                    raise ValueError("formal scenarios require inertial_pure_pursuit")
                if (
                    missile.heading_response_rate_per_s is None
                    or missile.max_turn_rate_deg_s is None
                ):
                    raise ValueError("formal scenarios require both inertial guidance parameters")
            else:
                if missile.guidance_model != "instantaneous_pure_pursuit":
                    raise ValueError(
                        "ablation scenarios require instantaneous_pure_pursuit"
                    )
                if (
                    missile.heading_response_rate_per_s is not None
                    or missile.max_turn_rate_deg_s is not None
                ):
                    raise ValueError("instantaneous ablation has no inertial parameters")
        return self


def load_scenario(path: str | Path) -> Scenario:
    """Load a YAML scenario and enforce the complete schema."""

    with Path(path).open(encoding="utf-8") as stream:
        return Scenario.model_validate(yaml.safe_load(stream))


def scenario_hash(
    scenario: Scenario,
    constants_path: str | Path = DEFAULT_CONSTANTS_PATH,
) -> str:
    """Hash normalized scenario content together with the constants source."""

    with Path(constants_path).open(encoding="utf-8") as stream:
        constants = yaml.safe_load(stream)
    payload = {
        "constants": constants,
        "scenario": scenario.model_dump(mode="json"),
    }
    normalized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
