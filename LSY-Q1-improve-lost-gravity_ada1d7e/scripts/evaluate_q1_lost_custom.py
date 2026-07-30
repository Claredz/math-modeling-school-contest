"""Judge-facing Q1 evaluator with editable missile position and bearing."""

from __future__ import annotations

import argparse
import json
from math import cos, radians, sin
from pathlib import Path

import numpy as np

from smoke_defense.q1_lost import make_custom_q1_scenario, solve_q1_lost_coupled
from smoke_defense.q1_lost_visualization import plot_timeline, plot_trajectory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Edit missile position directly with --missile-x/--missile-y, or "
            "use --distance/--direction-deg. Direction is the bearing from the "
            "ship to the missile at appearance."
        )
    )
    parser.add_argument("--missile-x", type=float)
    parser.add_argument("--missile-y", type=float)
    parser.add_argument("--distance", type=float, default=12000.0)
    parser.add_argument("--direction-deg", type=float, default=90.0)
    parser.add_argument("--initial-heading-error-deg", type=float, default=10.0)
    parser.add_argument("--heading-response-rate", type=float, default=1.0)
    parser.add_argument("--max-turn-rate-deg-s", type=float, default=5.0)
    parser.add_argument("--tracked-turn-time-constant-s", type=float, default=0.5)
    parser.add_argument("--lost-turn-decay-time-s", type=float, default=5.0)
    parser.add_argument("--reacquisition-confirm-s", type=float, default=1.0)
    parser.add_argument("--burst-start-s", type=float, default=5.5)
    parser.add_argument("--burst-stop-s", type=float, default=35.0)
    parser.add_argument("--burst-step-s", type=float, default=0.5)
    parser.add_argument("--integration-step-s", type=float, default=0.02)
    parser.add_argument("--final-time-s", type=float, default=150.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/q1_lost/custom"))
    return parser


def _position(args: argparse.Namespace) -> tuple[float, float]:
    supplied = args.missile_x is not None or args.missile_y is not None
    if supplied:
        if args.missile_x is None or args.missile_y is None:
            raise ValueError("--missile-x and --missile-y must be supplied together")
        return args.missile_x, args.missile_y
    angle = radians(args.direction_deg)
    return args.distance * cos(angle), args.distance * sin(angle)


def _solve(args: argparse.Namespace, position: tuple[float, float]):
    scenario = make_custom_q1_scenario(
        scenario_id="judge_custom_q1",
        missile_position_world_m=position,
        heading_response_rate_per_s=args.heading_response_rate,
        max_turn_rate_deg_s=args.max_turn_rate_deg_s,
    )
    coarse_times = tuple(
        np.arange(
            args.burst_start_s,
            args.burst_stop_s + 0.5 * args.burst_step_s,
            args.burst_step_s,
        )
    )
    coarse = solve_q1_lost_coupled(
        scenario,
        initial_heading_error_deg=args.initial_heading_error_deg,
        lost_turn_decay_time_s=args.lost_turn_decay_time_s,
        reacquisition_confirm_s=args.reacquisition_confirm_s,
        tracked_turn_time_constant_s=args.tracked_turn_time_constant_s,
        burst_times_s=coarse_times,
        final_time_s=args.final_time_s,
        time_step_s=args.integration_step_s,
    )
    if coarse.best_candidate is None:
        return coarse
    center = coarse.best_candidate.burst_time_s
    fine_step = args.burst_step_s / 10.0
    fine_times = tuple(
        np.arange(
            center - args.burst_step_s,
            center + args.burst_step_s + 0.5 * fine_step,
            fine_step,
        )
    )
    return solve_q1_lost_coupled(
        scenario,
        initial_heading_error_deg=args.initial_heading_error_deg,
        lost_turn_decay_time_s=args.lost_turn_decay_time_s,
        reacquisition_confirm_s=args.reacquisition_confirm_s,
        tracked_turn_time_constant_s=args.tracked_turn_time_constant_s,
        burst_times_s=fine_times,
        final_time_s=args.final_time_s,
        time_step_s=args.integration_step_s,
    )


def _candidate_payload(candidate):
    return {
        "takeoff_time_s": candidate.takeoff_time_s,
        "release_time_s": candidate.release_time_s,
        "burst_time_s": candidate.burst_time_s,
        "release_position_m": candidate.release_position_m.tolist(),
        "burst_center_m": candidate.burst_center_m.tolist(),
        "minimum_separation_m": candidate.trajectory.minimum_separation_m,
        "escaped_without_reacquisition": candidate.trajectory.escaped_without_reacquisition,
    }


def main() -> None:
    args = _parser().parse_args()
    position = _position(args)
    result = _solve(args, position)
    successful = [item for item in result.candidates if item.successful_defense]
    best = result.best_candidate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "missile_position_world_m": list(position),
        "model_parameters": {
            "initial_heading_error_deg": args.initial_heading_error_deg,
            "heading_response_rate_per_s": args.heading_response_rate,
            "max_turn_rate_deg_s": args.max_turn_rate_deg_s,
            "tracked_turn_time_constant_s": args.tracked_turn_time_constant_s,
            "lost_turn_decay_time_s": args.lost_turn_decay_time_s,
            "reacquisition_confirm_s": args.reacquisition_confirm_s,
            "burst_search_interval_s": [args.burst_start_s, args.burst_stop_s],
            "coarse_burst_step_s": args.burst_step_s,
            "refined_burst_step_s": args.burst_step_s / 10.0,
        },
        "defense_successful": result.feasible,
        "conclusion": (
            "存在可达投放方案，可使导弹永久失锁并飞离探测范围"
            if result.feasible
            else "在当前参数与搜索区间内未找到成功防御方案"
        ),
        "unique_optimum_on_refined_grid": result.unique_optimum_on_search_grid,
        "successful_condition_count": len(successful),
        "best_condition": _candidate_payload(best) if best else None,
        "successful_conditions": [_candidate_payload(item) for item in successful],
    }
    result_path = args.output_dir / "evaluation.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if best is not None:
        plot_trajectory(best, output_path=args.output_dir / "trajectory.png")
        plot_timeline(best, output_path=args.output_dir / "timeline.png")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"results and figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
