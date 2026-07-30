"""Run all approved Q1 geometries and write a traceable JSON result."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from smoke_defense.q1 import (
    solve_all_q1_sweeps,
    write_q1_markdown_summary,
    write_q1_sweep_result,
)

DEFAULT_OUTPUT = Path("results/q1/q1_sweep_results.json")
DEFAULT_SUMMARY_OUTPUT = Path("results/q1/README.md")
RANDOM_SEED = 20260730


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
    args = parser.parse_args()
    sweeps = solve_all_q1_sweeps()
    output = write_q1_sweep_result(
        sweeps,
        args.output,
        git_sha=_git_sha(),
        random_seed=RANDOM_SEED,
    )
    summary = write_q1_markdown_summary(sweeps, args.summary_output)
    print(
        f"wrote {len(sweeps)} Q1 geometry sweeps to {output} "
        f"and {summary}"
    )


if __name__ == "__main__":
    main()
