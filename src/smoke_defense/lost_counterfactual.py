"""Small, explicitly non-formal smoke-loss counterfactual for Q1 review."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

import numpy as np

from smoke_defense.coverage import single_smoke_gap
from smoke_defense.q1_rebuild import Q1CandidateDecision, Q1Problem


@dataclass(frozen=True)
class LostCounterfactualParameters:
    tau_t_s: float
    tau_l_s: float
    t_r_s: float

    def __post_init__(self) -> None:
        if min(self.tau_t_s, self.tau_l_s, self.t_r_s) <= 0:
            raise ValueError("loss and reacquisition parameters must be positive")


@dataclass(frozen=True)
class LostCounterfactualResult:
    label: str
    formal_baseline: bool
    lost: bool
    reacquired: bool
    hit: bool
    minimum_separation_m: float
    parameters: dict[str, float]


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def simulate_lost_counterfactual(
    problem: Q1Problem,
    candidate: Q1CandidateDecision,
    parameters: LostCounterfactualParameters,
    *,
    time_step_s: float = 0.05,
) -> LostCounterfactualResult:
    """Simulate only representative loss/reacquisition behavior.

    This intentionally does not replace the formal IPP detection trajectory. It is a
    small sensitivity witness with its own label and parameters.
    """

    scenario_missile = problem.scenario.missiles[0]
    initial = np.asarray(
        scenario_missile.initial_position_at_appearance_body_m
        or scenario_missile.initial_position_world_m,
        dtype=float,
    )
    time_s = float(scenario_missile.appearance_time_s)
    final_time = problem.detection.components[-1].end_s + 30.0
    position = initial.copy()
    heading = float(atan2(-position[1], -position[0]))
    turn_rate = 0.0
    mode = "tracked"
    visible_clock = 0.0
    lost = False
    reacquired = False
    hit = False
    minimum_separation = float("inf")
    while time_s <= final_time + 1e-12:
        ship = problem.ship.position(time_s)
        relative = ship - position
        distance = float(np.linalg.norm(relative))
        minimum_separation = min(minimum_separation, distance)
        if distance <= problem.constants.ship.effective_radius_m:
            hit = True
            break
        line_of_sight = atan2(relative[1], relative[0])
        smoke_gap = single_smoke_gap(
            ship,
            candidate.smoke.burst_center_m,
            candidate.smoke.radius(time_s),
            ship_radius_m=problem.constants.ship.effective_radius_m,
        )
        geometrically_visible = distance <= problem.constants.missile.detection_range_m
        occluded = smoke_gap <= 0.0
        if mode == "tracked" and (not geometrically_visible or occluded):
            mode = "lost"
            lost = True
            visible_clock = 0.0
        elif mode == "lost" and geometrically_visible and not occluded:
            visible_clock += time_step_s
            if visible_clock >= parameters.t_r_s:
                mode = "tracked"
                reacquired = True
                visible_clock = 0.0
        else:
            visible_clock = 0.0
        if mode == "tracked":
            commanded_rate = _wrap(line_of_sight - heading) / parameters.tau_t_s
            turn_rate += (commanded_rate - turn_rate) * time_step_s / parameters.tau_t_s
        else:
            turn_rate += -turn_rate * time_step_s / parameters.tau_l_s
        heading += turn_rate * time_step_s
        position += 320.0 * np.array([cos(heading), sin(heading)]) * time_step_s
        time_s += time_step_s
    return LostCounterfactualResult(
        label="experimental_counterfactual",
        formal_baseline=False,
        lost=lost,
        reacquired=reacquired,
        hit=hit,
        minimum_separation_m=minimum_separation,
        parameters={
            "tau_T_s": parameters.tau_t_s,
            "tau_L_s": parameters.tau_l_s,
            "T_R_s": parameters.t_r_s,
        },
    )
