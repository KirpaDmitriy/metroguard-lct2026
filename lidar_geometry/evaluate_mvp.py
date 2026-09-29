"""Reproducible sequence-level and ray-synthetic evaluation for the MVP."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import time

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import detect_hybrid
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.synthetic import SYNTHETIC_LIBRARY, inject_box
from lidar_geometry.temporal import TemporalConfirmer


ASSUMED_NORMAL = (
    "roundT_pressureGate_roundT",
    "roundT_doubleT",
    "roundT_squareT_pressureGate_squareT",
    "squareT_platform_squareT_switch",
    "doubleT_platform",
)
OBSTACLE_BAG = "doubleT_obstacle"


def bag_path(root: Path, name: str) -> Path:
    return root / name / f"{name}_0.db3"


def selected_frames(path: Path, every: int, maximum: int | None = None):
    emitted = 0
    for frame, (_, cloud) in enumerate(iter_bag_messages(path)):
        if frame % every:
            continue
        yield frame, cloud
        emitted += 1
        if maximum is not None and emitted >= maximum:
            break


def overlaps_distance(result, expected: float, tolerance: float = 2.0) -> bool:
    return any(
        item.distance_min_m - tolerance <= expected <= item.distance_max_m + tolerance
        for item in result.obstacles
    )


def evaluate(
    root: Path,
    every: int,
    synthetic_frames_per_bag: int,
    model: RiskModel | None = None,
) -> dict:
    config = DetectorConfig()
    detector = (
        (lambda cloud: detect_hybrid(cloud, model, config))
        if model is not None else (lambda cloud: detect_fast(cloud, config))
    )
    sequence_results = []
    latencies = []
    for name in (*ASSUMED_NORMAL, OBSTACLE_BAG):
        raw_counts = {"CLEAR": 0, "OBSTACLE": 0, "UNKNOWN": 0}
        counts = {"CLEAR": 0, "OBSTACLE": 0, "UNKNOWN": 0}
        distances = []
        confirmer = TemporalConfirmer(max_distance_step_m=3.5 * every)
        for frame, cloud in selected_frames(bag_path(root, name), every):
            started = time.perf_counter()
            result = detector(cloud)
            latencies.append((time.perf_counter() - started) * 1000)
            raw_counts[result.state] += 1
            temporal = confirmer.update(result)
            counts[temporal.state] += 1
            if temporal.obstacle and temporal.nearest_distance_m is not None:
                distances.append(temporal.nearest_distance_m)
        sequence_results.append({
            "bag": name,
            "assumed_normal_from_filename": name in ASSUMED_NORMAL,
            "sample_every": every,
            "raw_states": raw_counts,
            "temporal_states": counts,
            "alarm_rate": counts["OBSTACLE"] / max(1, sum(counts.values())),
            "nearest_alarm_m": min(distances, default=None),
        })

    synthetic_records = []
    distances = (20.0, 40.0, 60.0, 80.0, 100.0)
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        for frame, cloud in selected_frames(
            bag_path(root, name), max(1, every), synthetic_frames_per_bag
        ):
            base = detector(cloud)
            for prototype in SYNTHETIC_LIBRARY:
                for distance in distances:
                    obstacle = replace(prototype, distance_m=distance)
                    try:
                        injected = inject_box(
                            cloud, obstacle, config, seed=bag_index * 10000 + frame * 10 + round(distance)
                        )
                    except ValueError as error:
                        synthetic_records.append({
                            "bag": name, "frame": frame, "kind": obstacle.kind,
                            "distance_m": distance, "visible_points": 0,
                            "evaluable": False, "detected": False, "reason": str(error),
                        })
                        continue
                    result = detector(injected.cloud)
                    evaluable = injected.visible_points >= 3
                    synthetic_records.append({
                        "bag": name,
                        "frame": frame,
                        "kind": obstacle.kind,
                        "distance_m": distance,
                        "visible_points": injected.visible_points,
                        "evaluable": evaluable,
                        "detected": (
                            evaluable and result.state == "OBSTACLE"
                            and overlaps_distance(result, distance)
                        ),
                        "base_state": base.state,
                        "result_state": result.state,
                        "nearest_distance_m": result.nearest_distance_m,
                    })

    synthetic_summary = []
    for kind in sorted({item["kind"] for item in synthetic_records}):
        for distance in distances:
            rows = [
                item for item in synthetic_records
                if item["kind"] == kind and item["distance_m"] == distance
                and item["evaluable"]
            ]
            synthetic_summary.append({
                "kind": kind,
                "distance_m": distance,
                "evaluable": len(rows),
                "recall": sum(item["detected"] for item in rows) / len(rows) if rows else None,
                "median_visible_points": statistics.median(
                    item["visible_points"] for item in rows
                ) if rows else None,
            })

    return {
        "protocol": {
            "normal_is_weak_label": True,
            "normal_definition": "bag filename has no obstacle token; chat states at most two obstacles overall",
            "sample_every": every,
            "synthetic_method": "real background + ray/AABB intersection + nearer-return replacement",
            "synthetic_frames_per_bag": synthetic_frames_per_bag,
            "risk_model": model is not None,
        },
        "sequence_results": sequence_results,
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
            "mean": statistics.mean(latencies),
            "fps_from_median": 1000 / statistics.median(latencies),
        },
        "synthetic_summary": synthetic_summary,
        "synthetic_records": synthetic_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--synthetic-frames-per-bag", type=int, default=1)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("lidar_geometry/artifacts/mvp_evaluation.json"),
    )
    args = parser.parse_args()
    model = RiskModel.load(args.model) if args.model else None
    report = evaluate(args.dataset_root, args.every, args.synthetic_frames_per_bag, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "latency_ms": report["latency_ms"],
        "sequence_results": report["sequence_results"],
        "synthetic_summary": report["synthetic_summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
