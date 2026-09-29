from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

from lidar_geometry.detect_obstacles import DetectorConfig, Obstacle
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import TrackModel, detect_fast
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box


MODELS: tuple[TrackModel, ...] = ("greedy", "continuity")
DISTANCES_M = (20.0, 40.0, 60.0, 80.0, 100.0)


def run_detector(cloud, model: TrackModel):
    started = time.perf_counter()
    result = detect_fast(cloud, track_model=model)
    return result, (time.perf_counter() - started) * 1000


def exact_mask_benchmark(root: Path, frames_per_bag: int) -> dict:
    config = DetectorConfig()
    rows = []
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        for frame, cloud in selected_frames(bag_path(root, name), 50, frames_per_bag):
            for scenario_index, scenario in enumerate(SCENARIOS):
                for distance in DISTANCES_M:
                    target = replace(scenario.obstacle, distance_m=distance)
                    try:
                        injected = inject_box(
                            cloud,
                            target,
                            config,
                            seed=bag_index * 100_000 + frame * 100 + scenario_index,
                        )
                    except ValueError:
                        continue
                    if injected.visible_points < 3:
                        continue
                    row = {
                        "bag": name,
                        "frame": frame,
                        "scenario": scenario.name,
                        "expected_alarm": scenario.expected_alarm,
                        "distance_m": distance,
                        "visible_points": injected.visible_points,
                    }
                    for model in MODELS:
                        result = detect_fast(injected.cloud, config, track_model=model)
                        row[model] = any(
                            matches_target(component, target)
                            for component in result.obstacles
                        )
                    rows.append(row)
    positives = [row for row in rows if row["expected_alarm"]]
    negatives = [row for row in rows if not row["expected_alarm"]]
    return {
        "protocol": {
            "source_unit": "whole organizer bag",
            "frames_per_bag": frames_per_bag,
            "distances_m": list(DISTANCES_M),
            "target_attribution": "exact overlap with injected AABB",
            "minimum_visible_points": 3,
        },
        "positive_recall": {
            model: sum(row[model] for row in positives) / len(positives)
            for model in MODELS
        },
        "negative_specificity": {
            model: sum(not row[model] for row in negatives) / len(negatives)
            for model in MODELS
        },
        "rows": rows,
    }


def normal_bag_audit(root: Path) -> dict:
    bags = []
    latencies = {model: [] for model in MODELS}
    for name in ASSUMED_NORMAL:
        counts = {model: Counter() for model in MODELS}
        frames = 0
        for frame, cloud in selected_frames(bag_path(root, name), 1):
            order = MODELS if frame % 2 == 0 else tuple(reversed(MODELS))
            frames += 1
            for model in order:
                result, latency = run_detector(cloud, model)
                counts[model][result.state] += 1
                if frame:
                    latencies[model].append(latency)
        bags.append({
            "bag": name,
            "frames": frames,
            **{model: dict(counts[model]) for model in MODELS},
        })
        print(f"normal audit: {name} ({frames} frames)", flush=True)
    return {
        "protocol": {
            "source_unit": "whole organizer bag",
            "sampling": "every frame",
            "labels": "weak normal label from organizer filenames",
            "timing": "alternating detector order; first frame of each bag excluded",
        },
        "bags": bags,
        "latency_ms": {
            model: {
                "p50": statistics.median(values),
                "p95": sorted(values)[round(0.95 * (len(values) - 1))],
                "frames": len(values),
            }
            for model, values in latencies.items()
        },
    }


def overlaps_official_bounds(item: Obstacle, scenario) -> bool:
    target = scenario.obstacle
    return (
        item.lateral_max_m >= target.lateral_m - target.width_m / 2
        and item.lateral_min_m <= target.lateral_m + target.width_m / 2
        and item.height_max_m >= target.bottom_m
        and item.height_min_m <= target.bottom_m + target.height_m
    )


def official_pseudo_label_audit(
    bag: Path,
    motion_path: Path,
    first_world_m: float = 99.25,
    spacing_m: float = 100.0,
) -> dict:
    motion = json.loads(motion_path.read_text(encoding="utf-8"))["frames_detail"]
    scenarios = [{
        "scenario": scenario.name,
        "expected_alarm": scenario.expected_alarm,
        **{model: {"visible_frames": 0, "proposal_frames": 0} for model in MODELS},
    } for scenario in SCENARIOS[:10]]
    for frame, (_timestamp, cloud) in enumerate(iter_bag_messages(bag)):
        detections = {
            model: detect_fast(cloud, track_model=model) for model in MODELS
        }
        world_position = motion[frame]["cumulative_m"]
        for index, scenario in enumerate(SCENARIOS[:10]):
            expected_range = first_world_m + index * spacing_m - world_position
            if not 3.0 <= expected_range <= 120.0:
                continue
            for model in MODELS:
                scenarios[index][model]["visible_frames"] += 1
                matched = any(
                    abs(
                        (item.distance_min_m + item.distance_max_m) / 2
                        - expected_range
                    ) <= 2.0
                    and overlaps_official_bounds(item, scenario)
                    for item in detections[model].obstacles
                )
                scenarios[index][model]["proposal_frames"] += matched
        if frame and frame % 250 == 0:
            print(f"official audit: {frame} frames", flush=True)
    return {
        "protocol": {
            "blind": False,
            "label": "forensic pseudo-label using organizer order and approximate 100 m spacing",
            "first_world_m": first_world_m,
            "spacing_m": spacing_m,
            "range_tolerance_m": 2.0,
            "motion_source": str(motion_path),
        },
        "scenarios": scenarios,
    }


def summarize(exact: dict, normal: dict, official: dict) -> dict:
    return {
        "exact_positive_recall": exact["positive_recall"],
        "exact_negative_specificity": exact["negative_specificity"],
        "normal_raw_obstacle_frames": {
            model: sum(
                bag[model].get("OBSTACLE", 0) for bag in normal["bags"]
            )
            for model in MODELS
        },
        "official_positive_proposal_frames": {
            model: sum(
                row[model]["proposal_frames"]
                for row in official["scenarios"] if row["expected_alarm"]
            )
            for model in MODELS
        },
        "latency_p95_ms": {
            model: normal["latency_ms"][model]["p95"] for model in MODELS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--official-bag",
        type=Path,
        default=Path(
            "external_data/hackathon/synthetic_official/data/"
            "cloud_with_fake_obj/cloud_with_fake_obj_0.db3"
        ),
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=Path("lidar_geometry/artifacts/official_ego_motion.json"),
    )
    parser.add_argument("--frames-per-bag", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/continuity_path_exp066.json"),
    )
    args = parser.parse_args()
    exact = exact_mask_benchmark(args.dataset_root, args.frames_per_bag)
    print("exact-mask benchmark complete", flush=True)
    normal = normal_bag_audit(args.dataset_root)
    official = official_pseudo_label_audit(args.official_bag, args.motion)
    report = {
        "experiment": "EXP-066",
        "change": "rail centerline path selection only",
        "models": {
            "greedy": "frozen per-range predecessor selection",
            "continuity": "joint dynamic-programming rail-pair path",
        },
        "exact_mask": exact,
        "normal_bags": normal,
        "official_pseudo_label": official,
    }
    report["summary"] = summarize(exact, normal, official)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
