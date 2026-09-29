from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics

from lidar_geometry.detect_obstacles import DetectorConfig, Obstacle
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import CORE_HALF_WIDTH_M, detect_fast
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import SyntheticObject, inject_box


def target_alarm(
    components: tuple[Obstacle, ...],
    obstacle: SyntheticObject,
    model: RiskModel | None = None,
) -> bool:
    if model is not None:
        return any(
            matches_target(component, obstacle) and model.score(component) >= model.threshold
            for component in components
        )
    return any(
        matches_target(component, obstacle)
        and abs((component.lateral_min_m + component.lateral_max_m) / 2) <= CORE_HALF_WIDTH_M
        for component in components
    )


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = {}
    for record in records:
        groups.setdefault((record["scenario"], record["distance_m"]), []).append(record)
    summary = []
    for (name, distance), rows in sorted(groups.items()):
        evaluable = [row for row in rows if row["evaluable"]]
        summary.append({
            "scenario": name,
            "distance_m": distance,
            "expected_alarm": rows[0]["expected_alarm"],
            "attempted": len(rows),
            "evaluable": len(evaluable),
            "correct_rate": (
                sum(row["correct"] for row in evaluable) / len(evaluable)
                if evaluable else None
            ),
            "candidate_rate": (
                sum(row["target_candidate"] for row in evaluable) / len(evaluable)
                if evaluable else None
            ),
            "median_visible_points": (
                statistics.median(row["visible_points"] for row in evaluable)
                if evaluable else None
            ),
        })
    return summary


def evaluate(
    root: Path,
    every: int,
    frames_per_bag: int,
    distances: tuple[float, ...],
    model: RiskModel | None = None,
    guard_band_m: float = 0.0,
) -> dict:
    config = DetectorConfig(component_guard_band_m=guard_band_m)
    records = []
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        frames = selected_frames(bag_path(root, name), every, frames_per_bag)
        for frame, cloud in frames:
            for scenario_index, scenario in enumerate(SCENARIOS):
                for distance in distances:
                    obstacle = replace(scenario.obstacle, distance_m=distance)
                    try:
                        injected = inject_box(
                            cloud,
                            obstacle,
                            config,
                            seed=bag_index * 100_000 + frame * 100 + scenario_index,
                        )
                    except ValueError as error:
                        records.append({
                            "bag": name,
                            "frame": frame,
                            "scenario": scenario.name,
                            "distance_m": distance,
                            "expected_alarm": scenario.expected_alarm,
                            "visible_points": 0,
                            "evaluable": False,
                            "correct": False,
                            "reason": str(error),
                        })
                        continue
                    result = detect_fast(injected.cloud, config)
                    evaluable = injected.visible_points >= 3
                    matched = [
                        item for item in result.obstacles
                        if matches_target(item, obstacle)
                    ]
                    candidate = bool(matched)
                    alarm = target_alarm(result.obstacles, obstacle, model)
                    matched_scores = [
                        model.score(item) for item in matched
                        if model is not None
                    ]
                    records.append({
                        "bag": name,
                        "frame": frame,
                        "scenario": scenario.name,
                        "distance_m": distance,
                        "expected_alarm": scenario.expected_alarm,
                        "visible_points": injected.visible_points,
                        "evaluable": evaluable,
                        "target_candidate": candidate,
                        "target_alarm": alarm,
                        "target_score": max(matched_scores, default=None),
                        "matched_component": (
                            asdict(max(matched, key=lambda item: item.points))
                            if matched else None
                        ),
                        "correct": evaluable and alarm == scenario.expected_alarm,
                        "detector_state": result.state,
                    })
    return {
        "protocol": {
            "method": "ray-consistent AABB insertion into organizer backgrounds",
            "split_unit": "whole organizer bag",
            "backgrounds": list(ASSUMED_NORMAL),
            "sample_every": every,
            "frames_per_bag": frames_per_bag,
            "distances_m": list(distances),
            "minimum_visible_points": 3,
            "target_attribution": "distance and lateral overlap with inserted bounds",
            "risk_model": model is not None,
            "component_guard_band_m": guard_band_m,
        },
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "summary": summarize(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--frames-per-bag", type=int, default=2)
    parser.add_argument("--distances", type=float, nargs="+", default=(20, 40, 60))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--guard-band", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/scenario_benchmark.json"),
    )
    args = parser.parse_args()
    model = RiskModel.load(args.model) if args.model else None
    report = evaluate(
        args.dataset_root,
        args.every,
        args.frames_per_bag,
        tuple(args.distances),
        model,
        args.guard_band,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
