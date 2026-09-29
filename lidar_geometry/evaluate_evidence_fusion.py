from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.evaluate_tracker import count_episodes, target_only
from lidar_geometry.ego_motion import estimate_motion
from lidar_geometry.evidence_fusion import EvidenceFusion
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.scenario_catalog import SCENARIOS
from lidar_geometry.synthetic import inject_box


def evaluate_real(
    root: Path, every: int, model: RiskModel, world_coordinates: bool,
    physical_rules: bool, cable_only: bool,
) -> tuple[list[dict], list[float]]:
    rows = []
    latencies = []
    for name in ASSUMED_NORMAL:
        fusion = EvidenceFusion(
            model,
            max_distance_step_m=3.5 * every,
            world_coordinates=world_coordinates,
            physical_rules=physical_rules,
            cable_only=cable_only,
        )
        motion = estimate_motion(bag_path(root, name)) if world_coordinates else None
        states = []
        for frame, cloud in selected_frames(bag_path(root, name), every):
            started = time.perf_counter()
            cumulative_m = motion[frame]["cumulative_m"] if motion is not None else None
            decision = fusion.update(detect_fast(cloud), cumulative_m)
            latencies.append((time.perf_counter() - started) * 1000)
            states.append(decision.state)
        rows.append({
            "bag": name,
            "frames": len(states),
            "alarm_frames": states.count("OBSTACLE"),
            "alarm_episodes": count_episodes(states),
            "alarm_rate": states.count("OBSTACLE") / len(states),
        })
    return rows, latencies


def evaluate_synthetic(
    root: Path, model: RiskModel, frame_count: int, world_coordinates: bool,
    physical_rules: bool, cable_only: bool,
) -> list[dict]:
    rows = []
    scenarios = [scenario for scenario in SCENARIOS if scenario.expected_alarm]
    for bag_index, name in enumerate(ASSUMED_NORMAL[:3]):
        clouds = list(selected_frames(bag_path(root, name), 1, frame_count))
        for scenario_index, scenario in enumerate(scenarios):
            fusion = EvidenceFusion(
                model,
                world_coordinates=world_coordinates,
                physical_rules=physical_rules,
                cable_only=cable_only,
            )
            frames = []
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
                geometric = target_only(detect_fast(injected.cloud), obstacle)
                decision = fusion.update(
                    geometric, float(step) if world_coordinates else None
                )
                frames.append({
                    "frame": frame,
                    "distance_m": distance,
                    "visible_points": injected.visible_points,
                    "evaluable": injected.visible_points >= 3,
                    "detected": decision.obstacle,
                    "reason": decision.reason,
                })
            evaluable = [item for item in frames if item["evaluable"]]
            rows.append({
                "bag": name,
                "scenario": scenario.name,
                "evaluable_frames": len(evaluable),
                "detected_frames": sum(item["detected"] for item in evaluable),
                "sequence_detected": any(item["detected"] for item in evaluable),
                "first_stable_distance_m": next(
                    (item["distance_m"] for item in evaluable if item["detected"]),
                    None,
                ),
                "reasons": sorted({
                    item["reason"] for item in evaluable if item["detected"]
                }),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--synthetic-frames", type=int, default=12)
    parser.add_argument("--world-coordinates", action="store_true")
    parser.add_argument("--physical-rules", action="store_true")
    parser.add_argument("--cable-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/evidence_fusion_evaluation.json"),
    )
    args = parser.parse_args()
    model = RiskModel.load(args.model)
    real, latencies = evaluate_real(
        args.dataset_root, args.every, model, args.world_coordinates,
        args.physical_rules, args.cable_only,
    )
    synthetic = evaluate_synthetic(
        args.dataset_root, model, args.synthetic_frames, args.world_coordinates,
        args.physical_rules, args.cable_only,
    )
    report = {
        "protocol": {
            "normal_label_is_weak": True,
            "normal_split_unit": "whole organizer bag",
            "sample_every": args.every,
            "model": str(args.model),
            "high_lane": "model threshold plus two associated component hits",
            "critical_lane": "score >= 0.3 plus four associated compact or thin-vertical hits",
            "tracking_coordinates": "world" if args.world_coordinates else "range_lateral",
            "rescue_rule": (
                "cable_only" if args.cable_only else
                "physical" if args.physical_rules else "generic_critical_shape"
            ),
            "synthetic_target_attribution": "only matched inserted components can count as detections",
        },
        "real_sequences": real,
        "real_summary": {
            "frames": sum(item["frames"] for item in real),
            "alarm_frames": sum(item["alarm_frames"] for item in real),
            "alarm_episodes": sum(item["alarm_episodes"] for item in real),
        },
        "synthetic_sequences": synthetic,
        "synthetic_summary": {
            "sequences": len(synthetic),
            "detected_sequences": sum(item["sequence_detected"] for item in synthetic),
            "critical_lane_sequences": sum(
                "persistent_critical_geometry" in item["reasons"] for item in synthetic
            ),
        },
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "real_summary": report["real_summary"],
        "synthetic_summary": report["synthetic_summary"],
        "latency_ms": report["latency_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
