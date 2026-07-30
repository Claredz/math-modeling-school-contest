"""Export or verify the repository scenario JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smoke_defense.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "schema" / "scenario.schema.json"


def rendered_schema() -> str:
    return (
        json.dumps(
            Scenario.model_json_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = rendered_schema()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            raise SystemExit("scenario.schema.json is out of date")
        print("scenario.schema.json is current")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
