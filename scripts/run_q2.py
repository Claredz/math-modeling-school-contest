"""Run all formal and ablation Q2 scenarios with certified joint coverage."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from smoke_defense.q2 import (
    solve_all_q2_scenarios,
    write_q2_markdown_summary,
    write_q2_results,
)

DEFAULT_OUTPUT = Path("results/q2/q2_results.json")
DEFAULT_SUMMARY_OUTPUT = Path("results/q2/README.md")
RANDOM_SEED = 20260730
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )
    parser.add_argument("--spatial-tolerance-m", type=float, default=0.05)
    parser.add_argument("--time-tolerance-s", type=float, default=1e-3)
    parser.add_argument("--initial-polygon-sides", type=int, default=32)
    parser.add_argument("--maximum-polygon-sides", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    start = time.perf_counter()

    def report_progress(completed: int, total: int, scenario_id: str) -> None:
        print(
            f"[{completed}/{total}] completed {scenario_id}",
            flush=True,
        )

    formal_results, ablation_results = solve_all_q2_scenarios(
        workers=args.workers,
        progress=report_progress,
        spatial_tolerance_m=args.spatial_tolerance_m,
        time_tolerance_s=args.time_tolerance_s,
        initial_polygon_sides=args.initial_polygon_sides,
        maximum_polygon_sides=args.maximum_polygon_sides,
        random_seed=RANDOM_SEED,
    )
    output = write_q2_results(
        formal_results=formal_results,
        ablation_results=ablation_results,
        output_path=args.output,
        git_sha=_git_sha(),
        random_seed=RANDOM_SEED,
        spatial_tolerance_m=args.spatial_tolerance_m,
        time_tolerance_s=args.time_tolerance_s,
        initial_polygon_sides=args.initial_polygon_sides,
        maximum_polygon_sides=args.maximum_polygon_sides,
    )
    summary = write_q2_markdown_summary(
        formal_results=formal_results,
        ablation_results=ablation_results,
        output_path=args.summary_output,
    )
    elapsed_s = time.perf_counter() - start
    print(
        f"wrote {len(formal_results)} formal and "
        f"{len(ablation_results)} ablation Q2 scenarios to {output} "
        f"and {summary} in {elapsed_s:.3f} s"
    )


if __name__ == "__main__":
    main()
