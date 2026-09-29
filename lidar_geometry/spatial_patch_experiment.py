from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    fit_mlp,
    fold_metrics,
    linear,
    sample_weights,
    threshold_from_real,
)
from lidar_geometry.detect_obstacles import DetectorConfig, Obstacle
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.domain_guard import GRID, component_patch, track_coordinates
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.risk_model import component_features
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box


def build_cache(root: Path, output: Path) -> None:
    config = DetectorConfig()
    features, patches, labels, groups, records = [], [], [], [], []
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        for frame, cloud in selected_frames(bag_path(root, name), 20):
            detection = detect_fast(cloud, config)
            coordinates = track_coordinates(cloud, config)
            if coordinates is None:
                continue
            for item in detection.obstacles:
                features.append(component_features(item))
                patches.append(component_patch(coordinates, item))
                labels.append(0)
                groups.append(name)
                records.append({"bag": name, "frame": frame, "label": 0, "source": "real"})
            if frame % 100:
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
                    synthetic = detect_fast(injected.cloud, config)
                    matches = [
                        item for item in synthetic.obstacles
                        if matches_target(item, obstacle)
                    ]
                    if not matches:
                        continue
                    item = max(matches, key=lambda value: value.points)
                    synthetic_coordinates = track_coordinates(injected.cloud, config)
                    if synthetic_coordinates is None:
                        continue
                    features.append(component_features(item))
                    patches.append(component_patch(synthetic_coordinates, item))
                    labels.append(int(scenario.expected_alarm))
                    groups.append(name)
                    records.append({
                        "bag": name,
                        "frame": frame,
                        "label": int(scenario.expected_alarm),
                        "source": "synthetic",
                        "scenario": scenario.name,
                        "distance_m": target_distance,
                    })
        print(f"cached {name}: {len(records)} samples", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.asarray(features),
        patches=np.asarray(patches, dtype=np.float32),
        labels=np.asarray(labels),
        groups=np.asarray(groups),
        records_json=json.dumps(records, ensure_ascii=False),
    )


def convolution_descriptor(patches: np.ndarray) -> np.ndarray:
    kernels = np.asarray([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
        [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    ], dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        patches, (3, 3), axis=(-2, -1)
    )
    response = np.einsum("nchwij,kij->nckhw", windows, kernels, optimize=True)
    absolute = np.abs(response)
    convolution_stats = np.concatenate((
        absolute.mean(axis=(-2, -1)),
        absolute.std(axis=(-2, -1)),
        absolute.max(axis=(-2, -1)),
    ), axis=2).reshape(len(patches), -1)
    pooled = patches.reshape(len(patches), 3, 4, 4, 4, 4).mean(axis=(3, 5))
    return np.column_stack((pooled.reshape(len(patches), -1), convolution_stats))


def evaluate_models(cache: Path, output: Path) -> None:
    data = np.load(cache, allow_pickle=False)
    features = data["features"].astype(np.float64)
    patches = data["patches"]
    labels = data["labels"]
    groups = data["groups"]
    records = json.loads(str(data["records_json"]))
    spatial = convolution_descriptor(patches).astype(np.float64)
    models = {
        "component_linear": (
            features,
            lambda x, y, weight: fit_logistic_features(
                x, y, weight, linear, steps=1500, l2=0.06
            ),
        ),
        "spatial_cv_only": (
            spatial,
            lambda x, y, weight: fit_logistic_features(
                x, y, weight, linear, steps=1500, l2=0.06
            ),
        ),
        "component_plus_spatial": (
            np.column_stack((features, spatial)),
            lambda x, y, weight: fit_logistic_features(
                x, y, weight, linear, steps=1500, l2=0.06
            ),
        ),
        "spatial_cv_mlp": (
            spatial,
            lambda x, y, weight: fit_mlp(
                x, y, weight, hidden=8, steps=1200, l2=0.03
            ),
        ),
    }
    report = {
        "protocol": {
            "split": "leave-one-recording-out",
            "grid": [3, GRID, GRID],
            "channels": ["log_density", "max_height", "max_surface_residual"],
            "filters": ["sobel_x", "sobel_y", "laplacian", "horizontal", "vertical", "center_surround"],
            "intensity_used": False,
            "threshold": "99th percentile of training real-frame maximum scores",
        },
        "samples": len(labels),
        "models": {},
    }
    for name, (matrix, fitter) in models.items():
        folds = []
        for held_out in sorted(set(groups.tolist())):
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            test_records = [record for record, keep in zip(records, test) if keep]
            weight = sample_weights(labels[train], train_records)
            predictor = fitter(matrix[train], labels[train], weight)
            threshold = threshold_from_real(predictor.score(matrix[train]), train_records)
            folds.append({
                "held_out_bag": held_out,
                **fold_metrics(
                    predictor.score(matrix[test]), labels[test], test_records, threshold
                ),
            })
        report["models"][name] = {
            "features": matrix.shape[1],
            "folds": folds,
            "mean_positive_recall": float(np.mean([row["positive_recall"] for row in folds])),
            "worst_positive_recall": float(np.min([row["positive_recall"] for row in folds])),
            "mean_synthetic_negative_specificity": float(np.mean([
                row["synthetic_negative_specificity"] for row in folds
            ])),
            "mean_real_frame_alarm_rate": float(np.mean([
                row["real_frame_alarm_rate"] for row in folds
            ])),
            "worst_real_frame_alarm_rate": float(np.max([
                row["real_frame_alarm_rate"] for row in folds
            ])),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_comparison.json"),
    )
    args = parser.parse_args()
    if not args.cache.exists():
        build_cache(args.dataset_root, args.cache)
    evaluate_models(args.cache, args.output)


if __name__ == "__main__":
    main()
