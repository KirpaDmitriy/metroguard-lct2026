from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from lidar_geometry.detect_obstacles import DetectorConfig, Obstacle
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import SupportModel, detect_fast
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box


MODELS: tuple[SupportModel, ...] = ("frozen", "inverse_square")
DISTANCES_M = (20.0, 40.0, 60.0, 80.0, 100.0)


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
                    for mode in MODELS:
                        result = detect_fast(injected.cloud, config, support_model=mode)
                        row[mode] = any(
                            matches_target(component, target)
                            for component in result.obstacles
                        )
                    rows.append(row)

    by_distance = []
    for distance in DISTANCES_M:
        selected = [
            row for row in rows
            if row["distance_m"] == distance and row["expected_alarm"]
        ]
        by_distance.append({
            "distance_m": distance,
            "evaluable": len(selected),
            **{
                mode: sum(row[mode] for row in selected) / len(selected)
                for mode in MODELS
            },
        })
    negatives = [row for row in rows if not row["expected_alarm"]]
    negative_specificity = {
        mode: sum(not row[mode] for row in negatives) / len(negatives)
        for mode in MODELS
    }
    return {
        "protocol": {
            "source_unit": "whole organizer bag",
            "frames_per_bag": frames_per_bag,
            "distances_m": list(DISTANCES_M),
            "target_attribution": "exact overlap with injected AABB",
            "minimum_visible_points": 3,
        },
        "positive_recall_by_distance": by_distance,
        "negative_specificity": negative_specificity,
        "rows": rows,
    }


def normal_bag_audit(root: Path) -> dict:
    bags = []
    for name in ASSUMED_NORMAL:
        counts = {mode: Counter() for mode in MODELS}
        frames = 0
        for _frame, cloud in selected_frames(bag_path(root, name), 1):
            frames += 1
            for mode in MODELS:
                counts[mode][detect_fast(cloud, support_model=mode).state] += 1
        bags.append({
            "bag": name,
            "frames": frames,
            **{mode: dict(counts[mode]) for mode in MODELS},
        })
        print(f"normal audit: {name} ({frames} frames)", flush=True)
    return {
        "protocol": {
            "source_unit": "whole organizer bag",
            "sampling": "every frame",
            "labels": "weak normal label from organizer filenames",
        },
        "bags": bags,
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
        **{mode: {"visible_frames": 0, "proposal_frames": 0} for mode in MODELS},
    } for scenario in SCENARIOS[:10]]
    for frame, (_timestamp, cloud) in enumerate(iter_bag_messages(bag)):
        detections = {
            mode: detect_fast(cloud, support_model=mode) for mode in MODELS
        }
        world_position = motion[frame]["cumulative_m"]
        for index, scenario in enumerate(SCENARIOS[:10]):
            expected_range = first_world_m + index * spacing_m - world_position
            if not 3.0 <= expected_range <= 120.0:
                continue
            for mode in MODELS:
                scenarios[index][mode]["visible_frames"] += 1
                matched = any(
                    abs((item.distance_min_m + item.distance_max_m) / 2 - expected_range)
                    <= 2.0
                    and overlaps_official_bounds(item, scenario)
                    for item in detections[mode].obstacles
                )
                scenarios[index][mode]["proposal_frames"] += matched
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
    distant = [
        row for row in exact["positive_recall_by_distance"]
        if row["distance_m"] >= 60
    ]
    near = [
        row for row in exact["positive_recall_by_distance"]
        if row["distance_m"] <= 40
    ]
    normal_alarms = {
        mode: sum(bag[mode].get("OBSTACLE", 0) for bag in normal["bags"])
        for mode in MODELS
    }
    official_positive_proposals = {
        mode: sum(
            row[mode]["proposal_frames"]
            for row in official["scenarios"] if row["expected_alarm"]
        )
        for mode in MODELS
    }
    return {
        "distant_positive_recall": {
            mode: sum(row[mode] for row in distant) / len(distant) for mode in MODELS
        },
        "near_positive_recall": {
            mode: sum(row[mode] for row in near) / len(near) for mode in MODELS
        },
        "normal_raw_obstacle_frames": normal_alarms,
        "official_positive_proposal_frames": official_positive_proposals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--official-bag",
        type=Path,
        default=Path("external_data/hackathon/synthetic_official/data/cloud_with_fake_obj/cloud_with_fake_obj_0.db3"),
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
        default=Path("lidar_geometry/artifacts/range_support_comparison.json"),
    )
    args = parser.parse_args()
    exact = exact_mask_benchmark(args.dataset_root, args.frames_per_bag)
    print("exact-mask benchmark complete", flush=True)
    normal = normal_bag_audit(args.dataset_root)
    official = official_pseudo_label_audit(args.official_bag, args.motion)
    report = {
        "experiment": "EXP-064",
        "change": "component point-support curve only",
        "models": {
            "frozen": "max(3, round(6 * (30/r)^0.35))",
            "inverse_square": "max(3, round(6 * (30/r)^2))",
        },
        "exact_mask": exact,
        "normal_bags": normal,
        "official_pseudo_label": official,
    }
    report["summary"] = summarize(exact, normal, official)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
