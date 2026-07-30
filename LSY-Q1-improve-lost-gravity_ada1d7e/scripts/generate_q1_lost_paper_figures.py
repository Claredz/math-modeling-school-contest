"""Generate the Q1 lost-coupled feasibility map used in the paper."""

from __future__ import annotations

import csv
from math import cos, radians, sin
from pathlib import Path

import numpy as np

from smoke_defense.q1_lost import make_custom_q1_scenario, solve_q1_lost_coupled
from smoke_defense.q1_lost_visualization import plot_feasibility_heatmap

DIRECTIONS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
HEADING_ERRORS_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
DISTANCE_M = 12000.0
OUTPUT_DIR = Path("figures/q1_lost")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(HEADING_ERRORS_DEG), len(DIRECTIONS_DEG)))
    rows = []
    for row, heading_error in enumerate(HEADING_ERRORS_DEG):
        for column, direction in enumerate(DIRECTIONS_DEG):
            angle = radians(direction)
            position = (DISTANCE_M * cos(angle), DISTANCE_M * sin(angle))
            scenario = make_custom_q1_scenario(
                scenario_id=f"paper_d{int(direction)}_e{heading_error:g}",
                missile_position_world_m=position,
                heading_response_rate_per_s=1.0,
                max_turn_rate_deg_s=5.0,
            )
            result = solve_q1_lost_coupled(
                scenario,
                initial_heading_error_deg=heading_error,
                tracked_turn_time_constant_s=0.5,
                lost_turn_decay_time_s=10.0,
                reacquisition_confirm_s=1.0,
                burst_times_s=tuple(np.arange(12.0, 18.01, 0.5)),
                final_time_s=150.0,
                time_step_s=0.05,
            )
            matrix[row, column] = float(result.feasible)
            best = result.best_candidate
            rows.append(
                {
                    "direction_deg": direction,
                    "initial_heading_error_deg": heading_error,
                    "successful": result.feasible,
                    "best_burst_time_s": best.burst_time_s if best else "",
                    "minimum_separation_m": (
                        best.trajectory.minimum_separation_m if best else ""
                    ),
                }
            )
    plot_feasibility_heatmap(
        matrix,
        direction_degrees=DIRECTIONS_DEG,
        heading_errors_deg=HEADING_ERRORS_DEG,
        output_path=OUTPUT_DIR / "feasibility_heatmap.png",
    )
    with (OUTPUT_DIR / "feasibility_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote paper heatmap and matrix to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
