from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box
from lidar_geometry.temporal import ComponentTracker


def count_episodes(states: list[str]) -> int:
    return sum(
        state == "OBSTACLE" and (index == 0 or states[index - 1] != "OBSTACLE")
        for index, state in enumerate(states)
    )


def evaluate_real(root: Path, every: int, model: RiskModel) -> tuple[list[dict], list[float]]:
    results = []
    latencies = []
    for name in ASSUMED_NORMAL:
        tracker = ComponentTracker(max_distance_step_m=3.5 * every)
        raw_states = []
        tracked_states = []
        for _, cloud in selected_frames(bag_path(root, name), every):
            started = time.perf_counter()
            detection = filter_geometric_detection(detect_fast(cloud), model)
            latencies.append((time.perf_counter() - started) * 1000)
            raw_states.append(detection.state)
            tracked_states.append(tracker.update(detection).state)
        results.append({
            "bag": name,
            "frames": len(raw_states),
            "raw_alarm_frames": raw_states.count("OBSTACLE"),
            "tracked_alarm_frames": tracked_states.count("OBSTACLE"),
            "raw_episodes": count_episodes(raw_states),
            "tracked_episodes": count_episodes(tracked_states),
            "tracked_alarm_rate": tracked_states.count("OBSTACLE") / len(tracked_states),
        })
    return results, latencies


def target_only(detection, obstacle):
    matched = tuple(
        item for item in detection.obstacles if matches_target(item, obstacle)
    )
    return replace(
        detection,
        state="OBSTACLE" if matched else "UNKNOWN",
        obstacle=bool(matched),
        nearest_distance_m=min((item.distance_min_m for item in matched), default=None),
        obstacles=matched,
    )


def evaluate_synthetic(root: Path, model: RiskModel, frames_per_sequence: int) -> list[dict]:
    results = []
    positive_scenarios = [scenario for scenario in SCENARIOS if scenario.expected_alarm]
    for bag_index, name in enumerate(ASSUMED_NORMAL[:3]):
        clouds = list(selected_frames(bag_path(root, name), 1, frames_per_sequence))
        for scenario_index, scenario in enumerate(positive_scenarios):
            tracker = ComponentTracker()
            rows = []
            for step, (frame, cloud) in enumerate(clouds):
                distance = 40.0 - step
                obstacle = replace(scenario.obstacle, distance_m=distance)
                try:
                    injected = inject_box(
                        cloud,
                        obstacle,
                        seed=bag_index * 10_000 + scenario_index * 100 + step,
                    )
                except ValueError:
                    continue
                detection = filter_geometric_detection(detect_fast(injected.cloud), model)
                attributed = target_only(detection, obstacle)
                decision = tracker.update(attributed)
                rows.append({
                    "frame": frame,
                    "distance_m": distance,
                    "visible_points": injected.visible_points,
                    "evaluable": injected.visible_points >= 3,
                    "raw": attributed.state == "OBSTACLE",
                    "tracked": decision.state == "OBSTACLE",
                })
            evaluable = [row for row in rows if row["evaluable"]]
            results.append({
                "bag": name,
                "scenario": scenario.name,
                "evaluable_frames": len(evaluable),
                "raw_detected_frames": sum(row["raw"] for row in evaluable),
                "tracked_detected_frames": sum(row["tracked"] for row in evaluable),
                "sequence_detected": any(row["tracked"] for row in evaluable),
                "first_stable_distance_m": next(
                    (row["distance_m"] for row in evaluable if row["tracked"]),
                    None,
                ),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--synthetic-frames", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/component_tracker_evaluation.json"),
    )
    args = parser.parse_args()
    model = replace(RiskModel.load(args.model), threshold=args.threshold)
    real, latencies = evaluate_real(args.dataset_root, args.every, model)
    synthetic = evaluate_synthetic(args.dataset_root, model, args.synthetic_frames)
    report = {
        "protocol": {
            "normal_label_is_weak": True,
            "normal_split_unit": "whole organizer bag",
            "sample_every": args.every,
            "model": str(args.model),
            "threshold": args.threshold,
            "synthetic_sequence": "same scenario inserted into consecutive real frames at 40 m minus 1 m/frame",
            "synthetic_target_attribution": "only matched inserted components can count as detections",
        },
        "real_sequences": real,
        "real_summary": {
            "frames": sum(row["frames"] for row in real),
            "raw_alarm_frames": sum(row["raw_alarm_frames"] for row in real),
            "tracked_alarm_frames": sum(row["tracked_alarm_frames"] for row in real),
            "raw_episodes": sum(row["raw_episodes"] for row in real),
            "tracked_episodes": sum(row["tracked_episodes"] for row in real),
        },
        "synthetic_sequences": synthetic,
        "synthetic_summary": {
            "sequences": len(synthetic),
            "detected_sequences": sum(row["sequence_detected"] for row in synthetic),
        },
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "real_summary": report["real_summary"],
        "synthetic_summary": report["synthetic_summary"],
        "latency_ms": report["latency_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
