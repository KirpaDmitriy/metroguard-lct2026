from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

from lidar_geometry.ego_motion import estimate_motion
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.evaluate_tracker import count_episodes
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.temporal import WorldComponentTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/world_tracker_evaluation.json"),
    )
    args = parser.parse_args()
    model = RiskModel.load(args.model)
    thresholds = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    summaries = {threshold: [] for threshold in thresholds}
    latencies = []
    motion_summaries = []
    for name in ASSUMED_NORMAL:
        path = bag_path(args.dataset_root, name)
        motion = estimate_motion(path)
        motion_summaries.append({
            "bag": name,
            "distance_m": motion[-1]["cumulative_m"],
            "median_correlation": statistics.median(
                row["correlation"] for row in motion[1:]
            ),
        })
        trackers = {threshold: WorldComponentTracker() for threshold in thresholds}
        states = {threshold: [] for threshold in thresholds}
        for frame, cloud in selected_frames(path, args.every):
            started = time.perf_counter()
            geometric = detect_fast(cloud)
            latencies.append((time.perf_counter() - started) * 1000)
            for threshold in thresholds:
                scored = filter_geometric_detection(
                    geometric, replace(model, threshold=threshold)
                )
                decision = trackers[threshold].update(
                    scored, motion[frame]["cumulative_m"]
                )
                states[threshold].append(decision.state)
        for threshold in thresholds:
            values = states[threshold]
            summaries[threshold].append({
                "bag": name,
                "frames": len(values),
                "alarm_frames": values.count("OBSTACLE"),
                "alarm_episodes": count_episodes(values),
            })
    operating_points = []
    for threshold in thresholds:
        rows = summaries[threshold]
        operating_points.append({
            "threshold": threshold,
            "frames": sum(row["frames"] for row in rows),
            "alarm_frames": sum(row["alarm_frames"] for row in rows),
            "alarm_episodes": sum(row["alarm_episodes"] for row in rows),
            "by_bag": rows,
        })
    report = {
        "protocol": {
            "normal_label_is_weak": True,
            "sample_every": args.every,
            "motion": "wall-profile range correlation on every input frame",
            "tracking": "constant world coordinate after adding inferred ego motion",
        },
        "motion": motion_summaries,
        "operating_points": operating_points,
        "detector_latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "operating_points": operating_points,
        "detector_latency_ms": report["detector_latency_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
