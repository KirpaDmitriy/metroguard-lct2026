from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics

from lidar_geometry.ego_motion import EgoMotionEstimator
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.evaluate_tracker import count_episodes, target_only
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.scenario_catalog import SCENARIOS
from lidar_geometry.synthetic import inject_box
from lidar_geometry.temporal import ComponentTracker, WorldComponentTracker


VARIANTS = (
    "range_2of3",
    "range_3of5",
    "world_2of3",
    "world_3of5",
)


def trackers(distance_step_m: float = 3.5):
    return {
        "range_2of3": ComponentTracker(
            required_hits=2,
            max_missed_frames=2,
            max_distance_step_m=distance_step_m,
            confirmation_window_frames=3,
        ),
        "range_3of5": ComponentTracker(
            required_hits=3,
            max_missed_frames=4,
            max_distance_step_m=distance_step_m,
            confirmation_window_frames=5,
        ),
        "world_2of3": WorldComponentTracker(
            required_hits=2,
            max_missed_frames=2,
            confirmation_window_frames=3,
        ),
        "world_3of5": WorldComponentTracker(
            required_hits=3,
            max_missed_frames=4,
            confirmation_window_frames=5,
        ),
    }


def update_all(active, detection, cumulative_m: float):
    return {
        name: (
            tracker.update(detection, cumulative_m)
            if name.startswith("world_")
            else tracker.update(detection)
        )
        for name, tracker in active.items()
    }


def evaluate_normal(root: Path, model: RiskModel, every: int) -> list[dict]:
    rows = []
    for name in ASSUMED_NORMAL:
        active = trackers(distance_step_m=3.5 * every)
        states = {variant: [] for variant in VARIANTS}
        motion = EgoMotionEstimator()
        for _, cloud in selected_frames(bag_path(root, name), every):
            cumulative_m = motion.update(cloud).cumulative_m
            detection = filter_geometric_detection(detect_fast(cloud), model)
            decisions = update_all(active, detection, cumulative_m)
            for variant, decision in decisions.items():
                states[variant].append(decision.state)
        rows.append({
            "bag": name,
            "frames": len(next(iter(states.values()))),
            "variants": {
                variant: {
                    "alarm_frames": values.count("OBSTACLE"),
                    "alarm_episodes": count_episodes(values),
                }
                for variant, values in states.items()
            },
        })
    return rows


def evaluate_synthetic(root: Path, model: RiskModel, frames: int) -> list[dict]:
    rows = []
    positives = [scenario for scenario in SCENARIOS if scenario.expected_alarm]
    for bag_index, name in enumerate(ASSUMED_NORMAL[:3]):
        clouds = list(selected_frames(bag_path(root, name), 1, frames))
        for scenario_index, scenario in enumerate(positives):
            active = trackers()
            observations = {variant: [] for variant in VARIANTS}
            evaluable_frames = 0
            for step, (frame, cloud) in enumerate(clouds):
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
                attributed = target_only(
                    filter_geometric_detection(detect_fast(injected.cloud), model),
                    target,
                )
                evaluable = injected.visible_points >= 3
                evaluable_frames += evaluable
                decisions = update_all(active, attributed, float(step))
                for variant, decision in decisions.items():
                    observations[variant].append({
                        "frame": frame,
                        "step": step,
                        "distance_m": distance,
                        "evaluable": evaluable,
                        "confirmed": decision.obstacle,
                    })
            variants = {}
            for variant, values in observations.items():
                confirmations = [row for row in values if row["evaluable"] and row["confirmed"]]
                first = confirmations[0] if confirmations else None
                variants[variant] = {
                    "event_detected": bool(confirmations),
                    "confirmed_frames": len(confirmations),
                    "first_confirmation_step": first["step"] if first else None,
                    "first_confirmation_distance_m": first["distance_m"] if first else None,
                }
            rows.append({
                "bag": name,
                "scenario": scenario.name,
                "evaluable_frames": evaluable_frames,
                "variants": variants,
            })
    return rows


def summarize_normal(rows: list[dict]) -> dict:
    return {
        variant: {
            "frames": sum(row["frames"] for row in rows),
            "alarm_frames": sum(
                row["variants"][variant]["alarm_frames"] for row in rows
            ),
            "alarm_episodes": sum(
                row["variants"][variant]["alarm_episodes"] for row in rows
            ),
        }
        for variant in VARIANTS
    }


def summarize_synthetic(rows: list[dict]) -> dict:
    summary = {}
    for variant in VARIANTS:
        detected = [
            row for row in rows if row["variants"][variant]["event_detected"]
        ]
        distances = [
            row["variants"][variant]["first_confirmation_distance_m"]
            for row in detected
        ]
        baseline_delays = []
        for row in rows:
            baseline = row["variants"]["range_2of3"]["first_confirmation_step"]
            current = row["variants"][variant]["first_confirmation_step"]
            if baseline is not None and current is not None:
                baseline_delays.append(current - baseline)
        summary[variant] = {
            "events": len(rows),
            "detected_events": len(detected),
            "event_recall": len(detected) / len(rows),
            "median_first_confirmation_distance_m": (
                statistics.median(distances) if distances else None
            ),
            "median_delay_vs_range_2of3_frames": (
                statistics.median(baseline_delays) if baseline_delays else None
            ),
        }
    return summary


def reductions(candidate: dict, baseline: dict) -> dict:
    return {
        metric: (
            (baseline[metric] - candidate[metric]) / baseline[metric]
            if baseline[metric] else 0.0
        )
        for metric in ("alarm_frames", "alarm_episodes")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--synthetic-frames", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/temporal_window_exp065.json"),
    )
    args = parser.parse_args()
    model = replace(RiskModel.load(args.model), threshold=args.threshold)
    normal_rows = evaluate_normal(args.dataset_root, model, args.every)
    synthetic_rows = evaluate_synthetic(
        args.dataset_root, model, args.synthetic_frames
    )
    normal = summarize_normal(normal_rows)
    synthetic = summarize_synthetic(synthetic_rows)
    comparisons = {
        "range_3of5_vs_range_2of3": reductions(
            normal["range_3of5"], normal["range_2of3"]
        ),
        "world_3of5_vs_world_2of3": reductions(
            normal["world_3of5"], normal["world_2of3"]
        ),
    }
    report = {
        "experiment": "EXP-065",
        "protocol": {
            "normal_split_unit": "whole organizer bag",
            "normal_bags": list(ASSUMED_NORMAL),
            "sample_every": args.every,
            "model": str(args.model),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "threshold": args.threshold,
            "synthetic_sequence": "fixed seeded target at 40 m minus one metre per frame",
            "target_attribution": "exact overlap with inserted-object bounds",
            "frozen_components": ["geometry", "ranker", "threshold", "scenarios"],
        },
        "normal_summary": normal,
        "synthetic_summary": synthetic,
        "comparisons": comparisons,
        "normal_by_bag": normal_rows,
        "synthetic_sequences": synthetic_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "normal_summary": normal,
        "synthetic_summary": synthetic,
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
