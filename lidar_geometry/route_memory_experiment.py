from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.domain_guard import component_patch, track_coordinates
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.invariant_spatial_experiment import randomized_patches
from lidar_geometry.risk_model import component_features
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box


def change_score(reference: np.ndarray, current: np.ndarray) -> float:
    increase = np.maximum(current - reference, 0)
    weighted = 0.25 * increase[0] + 0.30 * increase[1] + 0.45 * increase[2]
    central = weighted[3:13, 3:13].ravel()
    keep = min(12, len(central))
    return float(np.partition(central, -keep)[-keep:].mean())


def aligned_change_score(reference: np.ndarray, current: np.ndarray) -> float:
    border = np.ones((16, 16), dtype=bool)
    border[4:12, 4:12] = False
    best = current
    best_error = np.inf
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            shifted = np.roll(current, (-row_shift, -column_shift), axis=(1, 2))
            error = np.abs(shifted[:, border] - reference[:, border]).mean()
            if error < best_error:
                best_error = error
                best = shifted
    return change_score(reference, best)


def coherent_aligned_change_score(reference: np.ndarray, current: np.ndarray) -> float:
    border = np.ones((16, 16), dtype=bool)
    border[4:12, 4:12] = False
    best = current
    best_error = np.inf
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            shifted = np.roll(current, (-row_shift, -column_shift), axis=(1, 2))
            error = np.abs(shifted[:, border] - reference[:, border]).mean()
            if error < best_error:
                best_error = error
                best = shifted
    increase = np.maximum(best - reference, 0)
    coherent = (
        np.sqrt(increase[0] * increase[2])
        + 0.5 * np.sqrt(increase[1] * increase[2])
    )
    central = coherent[3:13, 3:13].ravel()
    keep = min(12, len(central))
    return float(np.partition(central, -keep)[-keep:].mean())


def clearance_coherent_change_score(reference: np.ndarray, current: np.ndarray) -> float:
    border = np.ones((16, 16), dtype=bool)
    border[4:12, 4:12] = False
    best = current
    best_error = np.inf
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            shifted = np.roll(current, (-row_shift, -column_shift), axis=(1, 2))
            error = np.abs(shifted[:, border] - reference[:, border]).mean()
            if error < best_error:
                best_error = error
                best = shifted
    increase = np.maximum(best - reference, 0)
    coherent = (
        np.sqrt(increase[0] * increase[2])
        + 0.5 * np.sqrt(increase[1] * increase[2])
    )
    changed = coherent[3:13, 3:13] > 0.02
    changed_heights = best[1, 3:13, 3:13][changed]
    if len(changed_heights) and np.quantile(changed_heights, 0.1) >= 0.96:
        return 0.0
    central = coherent[3:13, 3:13].ravel()
    keep = min(12, len(central))
    return float(np.partition(central, -keep)[-keep:].mean())


def build_pairs(root: Path, output: Path) -> None:
    config = DetectorConfig()
    references = []
    currents = []
    labels = []
    groups = []
    records = []
    current_features = []
    clean_references = []
    clean_currents = []
    clean_groups = []
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        for frame, cloud in selected_frames(bag_path(root, name), 100):
            coordinates = track_coordinates(cloud, config)
            if coordinates is None:
                continue
            for scenario_index, scenario in enumerate(SCENARIOS):
                for target_distance in (20.0, 40.0, 60.0):
                    obstacle = replace(scenario.obstacle, distance_m=target_distance)
                    try:
                        injected = inject_box(
                            cloud,
                            obstacle,
                            config,
                            seed=bag_index * 100_000 + frame * 100 + scenario_index,
                        )
                    except ValueError:
                        continue
                    if injected.visible_points < 3:
                        continue
                    detection = detect_fast(injected.cloud, config)
                    matches = [
                        item for item in detection.obstacles
                        if matches_target(item, obstacle)
                    ]
                    if not matches:
                        continue
                    item = max(matches, key=lambda value: value.points)
                    injected_coordinates = track_coordinates(injected.cloud, config)
                    if injected_coordinates is None:
                        continue
                    reference = component_patch(coordinates, item)
                    current = component_patch(injected_coordinates, item)
                    references.append(reference)
                    currents.append(current)
                    current_features.append(component_features(item))
                    labels.append(int(scenario.expected_alarm))
                    groups.append(name)
                    records.append({
                        "bag": name,
                        "frame": frame,
                        "scenario": scenario.name,
                        "distance_m": target_distance,
                        "label": int(scenario.expected_alarm),
                    })
                    clean_references.append(reference)
                    clean_currents.append(randomized_patches(
                        reference[None],
                        seed=bag_index * 1_000_000 + frame * 1000 + scenario_index * 10,
                        mild=True,
                    )[0])
                    clean_groups.append(name)
        print(f"paired {name}: {len(records)} obstacle pairs", flush=True)
    np.savez_compressed(
        output,
        references=np.asarray(references, dtype=np.float32),
        currents=np.asarray(currents, dtype=np.float32),
        labels=np.asarray(labels),
        groups=np.asarray(groups),
        clean_references=np.asarray(clean_references, dtype=np.float32),
        clean_currents=np.asarray(clean_currents, dtype=np.float32),
        clean_groups=np.asarray(clean_groups),
        records_json=json.dumps(records, ensure_ascii=False),
        current_features=np.asarray(current_features, dtype=np.float64),
    )


def evaluate(cache: Path, output: Path) -> None:
    data = np.load(cache, allow_pickle=False)
    labels = data["labels"]
    groups = data["groups"]
    clean_groups = data["clean_groups"]
    records = json.loads(str(data["records_json"]))
    methods = {}
    for method_name, scorer, calibration in (
        ("unaligned", change_score, "q99"),
        ("outer_border_aligned", aligned_change_score, "q99"),
        ("outer_border_aligned_max_clean", aligned_change_score, "max"),
        ("coherent_aligned_max_clean", coherent_aligned_change_score, "max"),
        ("clearance_coherent_max_clean", clearance_coherent_change_score, "max"),
    ):
        scores = np.asarray([
            scorer(reference, current)
            for reference, current in zip(data["references"], data["currents"])
        ])
        clean_scores = np.asarray([
            scorer(reference, current)
            for reference, current in zip(data["clean_references"], data["clean_currents"])
        ])
        folds = []
        scenario_hits: dict[str, list[bool]] = {}
        for held_out in sorted(set(groups.tolist())):
            train_clean = clean_groups != held_out
            test = groups == held_out
            test_clean = clean_groups == held_out
            threshold = (
                float(clean_scores[train_clean].max())
                if calibration == "max"
                else float(np.quantile(clean_scores[train_clean], 0.99, method="higher"))
            )
            predicted = scores[test] >= threshold
            positive = labels[test] == 1
            negative = ~positive
            for index in np.flatnonzero(test & (labels == 1)):
                scenario_hits.setdefault(records[index]["scenario"], []).append(
                    bool(scores[index] >= threshold)
                )
            folds.append({
                "held_out_bag": held_out,
                "threshold": threshold,
                "positive_recall": float(predicted[positive].mean()),
                "negative_specificity": float(1 - predicted[negative].mean()),
                "clean_alarm_rate": float((clean_scores[test_clean] >= threshold).mean()),
                "positive_samples": int(positive.sum()),
                "negative_samples": int(negative.sum()),
            })
        methods[method_name] = {
            "folds": folds,
            "mean_positive_recall": float(np.mean([row["positive_recall"] for row in folds])),
            "worst_positive_recall": float(np.min([row["positive_recall"] for row in folds])),
            "mean_negative_specificity": float(np.mean([row["negative_specificity"] for row in folds])),
            "mean_clean_alarm_rate": float(np.mean([row["clean_alarm_rate"] for row in folds])),
            "worst_clean_alarm_rate": float(np.max([row["clean_alarm_rate"] for row in folds])),
            "scenario_recall": {
                name: {
                    "detected": int(sum(values)),
                    "samples": len(values),
                    "recall": float(np.mean(values)),
                }
                for name, values in sorted(scenario_hits.items())
            },
        }
    report = {
        "protocol": {
            "split": "leave-one-recording-out",
            "reference": "same rail-relative location from a clean traversal",
            "clean_variation": "mild density, dropout, z/residual noise, and registration shift",
            "threshold": "99th percentile or maximum training clean-change score, per method",
            "labels_used_for_threshold": False,
            "limitation": "controlled same-location reference/current pairs; repeated real traversal unavailable",
        },
        "pairs": len(labels),
        "methods": methods,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("lidar_geometry/artifacts/route_memory_pairs.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/route_memory_evaluation.json"),
    )
    args = parser.parse_args()
    if not args.cache.exists():
        build_pairs(args.dataset_root, args.cache)
    evaluate(args.cache, args.output)


if __name__ == "__main__":
    main()
