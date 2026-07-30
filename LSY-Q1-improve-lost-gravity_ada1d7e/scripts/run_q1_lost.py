"""Run the approved Q1 smoke-loss feasibility demonstration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from smoke_defense.q1_lost import solve_q1_lost_coupled
from smoke_defense.scenario_matrix import generate_q1_q3_matrix

DEFAULT_CONFIG = Path("configs/sweeps/lost_guidance.yaml")
DEFAULT_OUTPUT = Path("results/q1_lost/q1_lost_result.json")


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    demo = config["demonstration"]
    scenario = next(
        item
        for item in generate_q1_q3_matrix()
        if item.scenario_id == demo["scenario_id"]
    )
    burst_times = tuple(
        np.arange(
            demo["burst_time_start_s"],
            demo["burst_time_stop_s"] + 0.5 * demo["burst_time_step_s"],
            demo["burst_time_step_s"],
        )
    )
    result = solve_q1_lost_coupled(
        scenario,
        initial_heading_error_deg=demo["initial_heading_error_deg"],
        tracked_turn_time_constant_s=demo["tracked_turn_time_constant_s"],
        lost_turn_decay_time_s=demo["lost_turn_decay_time_s"],
        reacquisition_confirm_s=demo["reacquisition_confirm_s"],
        burst_times_s=burst_times,
        final_time_s=demo["final_time_s"],
        time_step_s=demo["integration_step_s"],
    )
    best = result.best_candidate
    payload = {
        "git_sha": _git_sha(),
        "model_version": config["model_version"],
        "scenario_id": result.scenario_id,
        "assumption_ids": ["A-023", "A-024", "A-025", "A-026"],
        "initial_heading_error_deg": result.initial_heading_error_deg,
        "feasible": result.feasible,
        "unique_optimum_on_search_grid": result.unique_optimum_on_search_grid,
        "candidate_count": len(result.candidates),
        "best_candidate": None,
    }
    if best is not None:
        payload["best_candidate"] = {
            "successful_defense": best.successful_defense,
            "takeoff_time_s": best.takeoff_time_s,
            "release_time_s": best.release_time_s,
            "burst_time_s": best.burst_time_s,
            "release_position_m": best.release_position_m.tolist(),
            "burst_center_m": best.burst_center_m.tolist(),
            "flight_distance_m": best.flight_distance_m,
            "minimum_separation_m": best.trajectory.minimum_separation_m,
            "hit_time_s": best.trajectory.hit_time_s,
            "escaped_without_reacquisition": (
                best.trajectory.escaped_without_reacquisition
            ),
            "events": [
                {"time_s": event.time_s, "kind": event.kind}
                for event in best.trajectory.events
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Q1 lost-coupled feasible={result.feasible}, "
        f"unique_on_grid={result.unique_optimum_on_search_grid}; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
