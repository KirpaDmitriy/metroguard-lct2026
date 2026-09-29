from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.evaluate_tracker import target_only
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.scenario_catalog import SCENARIOS
from lidar_geometry.synthetic import inject_box
from lidar_geometry.temporal import WorldComponentTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path(
        "lidar_geometry/artifacts/world_tracker_synthetic.json"
    ))
    args = parser.parse_args()
    model = RiskModel.load(args.model)
    thresholds = (0.8, 0.9)
    rows = []
    positives = [scenario for scenario in SCENARIOS if scenario.expected_alarm]
    for bag_index, name in enumerate(ASSUMED_NORMAL[:3]):
        clouds = list(selected_frames(bag_path(args.dataset_root, name), 1, args.frames))
        for scenario_index, scenario in enumerate(positives):
            trackers = {threshold: WorldComponentTracker() for threshold in thresholds}
            detected = {threshold: False for threshold in thresholds}
            first_range = {threshold: None for threshold in thresholds}
            for step, (_frame, cloud) in enumerate(clouds):
                distance = 40.0 - step
                target = replace(scenario.obstacle, distance_m=distance)
                try:
                    injected = inject_box(
                        cloud,
                        target,
                        seed=bag_index * 10_000 + scenario_index * 100 + step,
                    )
                except ValueError:
                    continue
                geometric = detect_fast(injected.cloud)
                for threshold in thresholds:
                    filtered = filter_geometric_detection(
                        geometric, replace(model, threshold=threshold)
                    )
                    attributed = target_only(filtered, target)
                    decision = trackers[threshold].update(attributed, float(step))
                    if decision.obstacle:
                        detected[threshold] = True
                        if first_range[threshold] is None:
                            first_range[threshold] = distance
            rows.append({
                "bag": name,
                "scenario": scenario.name,
                "detected": {str(key): value for key, value in detected.items()},
                "first_detection_range_m": {
                    str(key): value for key, value in first_range.items()
                },
            })
    summary = {
        str(threshold): sum(row["detected"][str(threshold)] for row in rows)
        for threshold in thresholds
    }
    report = {
        "protocol": {
            "sequences": len(rows),
            "target_attribution": "strict overlap with inserted-object bounds",
            "motion": "one metre per synthetic frame, matching injected range change",
        },
        "detected_sequences": summary,
        "sequences": rows,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "detected_sequences": summary}, indent=2))


if __name__ == "__main__":
    main()
