from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from lidar_geometry.detect_obstacles import DetectorConfig, Obstacle
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box
from lidar_geometry.uncertainty_clearance import (
    apply_uncertainty_gate,
    center_uncertainty_m,
    confidently_in_clearance,
)


def detect_pair(cloud, model: RiskModel, config: DetectorConfig):
    profiles = []
    geometric = detect_fast(cloud, config, track_profile_out=profiles)
    baseline = filter_geometric_detection(geometric, model)
    return baseline, apply_uncertainty_gate(baseline, profiles[0], config), profiles[0]


def evaluate_normal(root: Path, model: RiskModel, config: DetectorConfig, every: int):
    rows = []
    for name in ASSUMED_NORMAL:
        states = []
        for _frame, cloud in selected_frames(bag_path(root, name), every):
            baseline, gated, _profile = detect_pair(cloud, model, config)
            states.append((baseline.state, gated.state))
        rows.append({
            "bag": name,
            "frames": len(states),
            "baseline_alarm_frames": sum(before == "OBSTACLE" for before, _ in states),
            "gated_alarm_frames": sum(after == "OBSTACLE" for _, after in states),
            "boundary_unknown_frames": sum(
                before == "OBSTACLE" and after == "UNKNOWN" for before, after in states
            ),
        })
    return rows


def evaluate_synthetic(root, model, config, every, frames_per_bag, distances):
    records = []
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        for frame, cloud in selected_frames(
            bag_path(root, name), every, frames_per_bag
        ):
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
                            "bag": name, "frame": frame, "scenario": scenario.name,
                            "distance_m": distance,
                            "expected_alarm": scenario.expected_alarm,
                            "evaluable": False, "reason": str(error),
                        })
                        continue
                    baseline, _gated, profile = detect_pair(injected.cloud, model, config)
                    matched = [
                        item for item in baseline.obstacles if matches_target(item, obstacle)
                    ]
                    confident = [
                        item for item in matched
                        if confidently_in_clearance(
                            item,
                            center_uncertainty_m(
                                profile,
                                (item.distance_min_m + item.distance_max_m) / 2,
                                config,
                            ),
                            config.half_width_m,
                        )
                    ]
                    evaluable = injected.visible_points >= 3
                    baseline_alarm = bool(matched)
                    gated_alarm = bool(confident)
                    records.append({
                        "bag": name,
                        "frame": frame,
                        "scenario": scenario.name,
                        "distance_m": distance,
                        "expected_alarm": scenario.expected_alarm,
                        "visible_points": injected.visible_points,
                        "evaluable": evaluable,
                        "baseline_alarm": baseline_alarm,
                        "gated_alarm": gated_alarm,
                        "baseline_correct": evaluable and baseline_alarm == scenario.expected_alarm,
                        "gated_correct": evaluable and gated_alarm == scenario.expected_alarm,
                        "matched_components": [asdict(item) for item in matched],
                    })
    return records


def rate(rows, prediction, expected):
    selected = [
        row for row in rows
        if row.get("evaluable") and row["expected_alarm"] is expected
    ]
    if not selected:
        return None
    correct = sum(
        row[prediction] if expected else not row[prediction]
        for row in selected
    )
    return correct / len(selected)


def boundary_cases(config):
    definitions = (
        ("deep_inside", -0.20, 0.20, 0.20, True),
        ("edge_with_stable_fit", 1.03, 1.04, 0.025, False),
        ("edge_with_sparse_fit", 0.90, 1.04, 0.20, False),
        ("wide_intrusion_with_sparse_fit", -0.90, 1.04, 0.20, True),
    )
    rows = []
    for name, lateral_min, lateral_max, uncertainty, expected in definitions:
        item = Obstacle(
            10, 3, 19.8, 20.2, lateral_min, lateral_max,
            0.1, 0.3, 0.3, 0.2, 10.0,
        )
        actual = confidently_in_clearance(item, uncertainty, config.half_width_m)
        rows.append({
            "case": name,
            "center_uncertainty_m": uncertainty,
            "expected_confident": expected,
            "actual_confident": actual,
            "correct": actual == expected,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--model", type=Path,
        default=Path("lidar_geometry/models/risk_model_g4_cost_sensitive.json"),
    )
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--synthetic-every", type=int, default=50)
    parser.add_argument("--frames-per-bag", type=int, default=2)
    parser.add_argument("--distances", type=float, nargs="+", default=(20, 40, 60))
    parser.add_argument(
        "--output", type=Path,
        default=Path("lidar_geometry/artifacts/uncertainty_clearance.json"),
    )
    args = parser.parse_args()
    model = RiskModel.load(args.model)
    config = DetectorConfig(component_guard_band_m=0.30)
    normal = evaluate_normal(args.dataset_root, model, config, args.every)
    synthetic = evaluate_synthetic(
        args.dataset_root, model, config, args.synthetic_every,
        args.frames_per_bag, tuple(args.distances),
    )
    cases = boundary_cases(config)
    normal_frames = sum(row["frames"] for row in normal)
    baseline_normal = sum(row["baseline_alarm_frames"] for row in normal)
    gated_normal = sum(row["gated_alarm_frames"] for row in normal)
    metrics = {
        "baseline_positive_recall": rate(synthetic, "baseline_alarm", True),
        "gated_positive_recall": rate(synthetic, "gated_alarm", True),
        "baseline_negative_specificity": rate(synthetic, "baseline_alarm", False),
        "gated_negative_specificity": rate(synthetic, "gated_alarm", False),
        "baseline_normal_alarm_rate": baseline_normal / normal_frames,
        "gated_normal_alarm_rate": gated_normal / normal_frames,
        "normal_alarm_frames_removed": baseline_normal - gated_normal,
        "boundary_cases_correct": sum(row["correct"] for row in cases),
        "boundary_cases": len(cases),
    }
    report = {
        "protocol": {
            "model": str(args.model),
            "half_clearance_m": config.half_width_m,
            "component_guard_band_m": config.component_guard_band_m,
            "uncertainty_floor_m": config.path_cell_m / 2,
            "normal_split_unit": "whole organizer bag",
            "normal_sample_every": args.every,
            "synthetic_background_split_unit": "whole organizer bag",
            "synthetic_frames_per_bag": args.frames_per_bag,
            "distances_m": args.distances,
            "minimum_visible_points": 3,
            "target_attribution": "exact injected distance and lateral bounds",
        },
        "metrics": metrics,
        "normal_bags": normal,
        "boundary_cases": cases,
        "synthetic_records": synthetic,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
