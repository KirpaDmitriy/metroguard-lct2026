#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "registry.json"

REQUIRED_RUN = {
    "id",
    "title",
    "status",
    "created",
    "hypothesis",
    "change",
    "data",
    "command",
    "metrics",
    "decision",
    "conclusion",
    "artifacts",
}
STATUSES = {"planned", "running", "complete", "failed"}
DECISIONS = {"accept", "reject", "inconclusive", "reject_as_submission_candidate"}


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def check() -> None:
    runs = load_json(REGISTRY)
    errors = []

    run_ids = [run.get("id") for run in runs]
    if len(run_ids) != len(set(run_ids)):
        errors.append("duplicate experiment id")

    for run in runs:
        missing = REQUIRED_RUN - run.keys()
        if missing:
            errors.append(f"{run.get('id', '<unknown>')}: missing {sorted(missing)}")
        if run.get("status") not in STATUSES:
            errors.append(f"{run.get('id')}: invalid status {run.get('status')!r}")
        if run.get("decision") not in DECISIONS:
            errors.append(f"{run.get('id')}: invalid decision {run.get('decision')!r}")
        data = run.get("data", {})
        for key in ("sources", "split_unit", "leakage_check", "blind"):
            if key not in data:
                errors.append(f"{run.get('id')}: data.{key} is required")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(runs)} experiments")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check",))
    parser.parse_args()
    check()


if __name__ == "__main__":
    main()
