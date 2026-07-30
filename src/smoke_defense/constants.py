"""Load immutable constants stated by the problem."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    """Base model that rejects unknown constants and prevents mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ShipConstants(FrozenModel):
    speed_mps: float = Field(gt=0)
    effective_radius_m: float = Field(gt=0)


class UavConstants(FrozenModel):
    speed_mps: float = Field(gt=0)
    operation_radius_m: float = Field(gt=0)
    max_payload_count: int = Field(gt=0)


class MissileConstants(FrozenModel):
    nominal_speed_mps: float = Field(gt=0)
    detection_range_m: float = Field(gt=0)
    field_of_view_half_angle_deg: float = Field(gt=0, le=180)


class CountermeasureConstants(FrozenModel):
    release_response_min_s: float = Field(ge=0)
    minimum_release_interval_s: float = Field(ge=0)
    detonation_delay_s: float = Field(ge=0)


class SmokeConstants(FrozenModel):
    maximum_radius_m: float = Field(gt=0)
    hold_duration_s: float = Field(ge=0)
    decay_duration_s: float = Field(gt=0)


class ProblemConstants(FrozenModel):
    constants_version: str
    ship: ShipConstants
    uav: UavConstants
    missile: MissileConstants
    countermeasure: CountermeasureConstants
    smoke: SmokeConstants


DEFAULT_CONSTANTS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "constants.yaml"
)


def load_problem_constants(path: str | Path | None = None) -> ProblemConstants:
    """Load and validate the single source of public problem constants."""

    constants_path = Path(path) if path is not None else DEFAULT_CONSTANTS_PATH
    with constants_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return ProblemConstants.model_validate(raw)
