"""Validate the approved formal and ablation scenario matrices."""

from __future__ import annotations

import argparse

from smoke_defense.scenario import scenario_hash
from smoke_defense.scenario_matrix import (
    generate_instantaneous_ablation_matrix,
    generate_q1_q3_matrix,
    generate_q1_rebuild_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    formal = generate_q1_q3_matrix()
    rebuild = generate_q1_rebuild_matrix()
    ablation = generate_instantaneous_ablation_matrix()
    if len(formal) != 144 or len(rebuild) != 16 or len(ablation) != 16:
        raise SystemExit("unexpected scenario-matrix size")
    hashes = {
        scenario_hash(scene)
        for scene in (*formal, *rebuild, *ablation)
    }
    if len(hashes) != 176:
        raise SystemExit("scenario hashes are not unique")
    print(
        f"validated {len(rebuild)} formal rebuild, "
        f"{len(formal)} inertial counterfactual-compatible, "
        f"and {len(ablation)} legacy ablation scenarios"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
