from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_geometry.classical_hybrid_experiment import extra_trees
from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    linear,
    sample_weights,
    threshold_from_real,
)
from lidar_geometry.complex_synthetic import inject_composite, scenario_library
from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.risk_model import component_features
from lidar_geometry.scenario_catalog import matches_target


def build_cache(root: Path, output: Path) -> None:
    config = DetectorConfig()
    features = []
    labels = []
    groups = []
    records = []
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        for frame, cloud in selected_frames(bag_path(root, name), 100):
            for distance in (20.0, 40.0, 60.0):
                for scenario_index, scenario in enumerate(scenario_library(distance)):
                    try:
                        injected = inject_composite(
                            cloud, scenario, config,
                            seed=bag_index * 100_000 + frame * 100 + scenario_index,
                        )
                    except ValueError:
                        continue
                    if injected.visible_points < 3:
                        continue
                    detection = detect_fast(injected.cloud, config)
                    matches = [
                        item for item in detection.obstacles
                        if matches_target(item, scenario.envelope)
                    ]
                    proposed = bool(matches)
                    item = max(matches, key=lambda value: value.points) if matches else None
                    features.append(
                        component_features(item) if item is not None else np.zeros(13)
                    )
                    labels.append(int(scenario.expected_alarm))
                    groups.append(name)
                    records.append({
                        "bag": name,
                        "frame": frame,
                        "scenario": scenario.name,
                        "distance_m": distance,
                        "label": int(scenario.expected_alarm),
                        "visible_points": injected.visible_points,
                        "proposed": proposed,
                    })
        print(f"built complex shapes for {name}: {len(records)} cases", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.asarray(features, dtype=np.float64),
        labels=np.asarray(labels),
        groups=np.asarray(groups),
        records_json=json.dumps(records, ensure_ascii=False),
    )


def metrics(
    scores: np.ndarray, labels: np.ndarray, records: list[dict], threshold: float
) -> dict:
    proposed = np.asarray([record["proposed"] for record in records])
    predicted = proposed & (scores >= threshold)
    positive = labels == 1
    negative = ~positive
    by_scenario = {}
    for scenario in sorted({record["scenario"] for record in records}):
        mask = np.asarray([record["scenario"] == scenario for record in records])
        if labels[mask][0]:
            by_scenario[scenario] = float(predicted[mask].mean())
        else:
            by_scenario[scenario] = float((~predicted[mask]).mean())
    return {
        "positive_recall": float(predicted[positive].mean()),
        "proposal_recall": float(proposed[positive].mean()),
        "negative_specificity": (
            float((~predicted[negative]).mean()) if negative.any() else None
        ),
        "scenario_score": by_scenario,
    }


def evaluate(box_cache: Path, shape_cache: Path, output: Path) -> dict:
    box = np.load(box_cache, allow_pickle=False)
    shape = np.load(shape_cache, allow_pickle=False)
    train_x = box["features"].astype(np.float64)
    train_y = box["labels"]
    train_groups = box["groups"]
    train_records = json.loads(str(box["records_json"]))
    shape_x = shape["features"]
    shape_y = shape["labels"]
    shape_groups = shape["groups"]
    shape_records = json.loads(str(shape["records_json"]))
    models = {name: [] for name in ("component_linear", "extra_trees", "tree_weighted_75")}
    for held_out in sorted(set(train_groups.tolist())):
        train = train_groups != held_out
        test = shape_groups == held_out
        records = [record for record, keep in zip(train_records, train) if keep]
        test_records = [record for record, keep in zip(shape_records, test) if keep]
        weight = sample_weights(train_y[train], records)
        linear_model = fit_logistic_features(
            train_x[train], train_y[train], weight, linear, steps=1500, l2=0.06
        )
        tree_model = extra_trees()
        tree_model.fit(train_x[train], train_y[train], sample_weight=weight)
        linear_train = linear_model.score(train_x[train])
        tree_train = tree_model.predict_proba(train_x[train])[:, 1]
        linear_test = linear_model.score(shape_x[test])
        tree_test = tree_model.predict_proba(shape_x[test])[:, 1]
        score_sets = {
            "component_linear": (linear_train, linear_test),
            "extra_trees": (tree_train, tree_test),
            "tree_weighted_75": (
                0.25 * linear_train + 0.75 * tree_train,
                0.25 * linear_test + 0.75 * tree_test,
            ),
        }
        for name, (calibration, scores) in score_sets.items():
            threshold = threshold_from_real(calibration, records)
            models[name].append({
                "held_out_bag": held_out,
                "threshold": threshold,
                **metrics(scores, shape_y[test], test_records, threshold),
            })
    summary = {}
    for name, folds in models.items():
        summary[name] = {
            "folds": folds,
            "mean_positive_recall": float(np.mean([row["positive_recall"] for row in folds])),
            "worst_positive_recall": float(np.min([row["positive_recall"] for row in folds])),
            "mean_proposal_recall": float(np.mean([row["proposal_recall"] for row in folds])),
            "mean_negative_specificity": float(np.mean([row["negative_specificity"] for row in folds])),
        }
    augmented_folds = []
    for held_out in sorted(set(train_groups.tolist())):
        box_train = train_groups != held_out
        shape_train = (shape_groups != held_out) & np.asarray([
            record["proposed"] for record in shape_records
        ])
        shape_test = shape_groups == held_out
        combined_x = np.vstack((train_x[box_train], shape_x[shape_train]))
        combined_y = np.concatenate((train_y[box_train], shape_y[shape_train]))
        combined_records = [
            record for record, keep in zip(train_records, box_train) if keep
        ] + [
            record for record, keep in zip(shape_records, shape_train) if keep
        ]
        weight = sample_weights(combined_y, combined_records)
        linear_model = fit_logistic_features(
            combined_x, combined_y, weight, linear, steps=1500, l2=0.06
        )
        tree_model = extra_trees()
        tree_model.fit(combined_x, combined_y, sample_weight=weight)
        calibration_x = train_x[box_train]
        calibration_records = [
            record for record, keep in zip(train_records, box_train) if keep
        ]
        calibration = (
            0.25 * linear_model.score(calibration_x)
            + 0.75 * tree_model.predict_proba(calibration_x)[:, 1]
        )
        test_scores = (
            0.25 * linear_model.score(shape_x[shape_test])
            + 0.75 * tree_model.predict_proba(shape_x[shape_test])[:, 1]
        )
        threshold = threshold_from_real(calibration, calibration_records)
        test_records = [
            record for record, keep in zip(shape_records, shape_test) if keep
        ]
        augmented_folds.append({
            "held_out_bag": held_out,
            "threshold": threshold,
            **metrics(test_scores, shape_y[shape_test], test_records, threshold),
        })
    summary["complex_augmented_tree_weighted_75"] = {
        "folds": augmented_folds,
        "mean_positive_recall": float(np.mean([
            row["positive_recall"] for row in augmented_folds
        ])),
        "worst_positive_recall": float(np.min([
            row["positive_recall"] for row in augmented_folds
        ])),
        "mean_proposal_recall": float(np.mean([
            row["proposal_recall"] for row in augmented_folds
        ])),
        "mean_negative_specificity": float(np.mean([
            row["negative_specificity"] for row in augmented_folds
        ])),
    }
    positive_shapes = sorted({
        record["scenario"] for record in shape_records if record["label"] == 1
    })
    unseen_shape_folds = []
    proposed_shape = np.asarray([record["proposed"] for record in shape_records])
    scenario_shape = np.asarray([record["scenario"] for record in shape_records])
    for held_out in sorted(set(train_groups.tolist())):
        for held_out_shape in positive_shapes:
            box_train = train_groups != held_out
            shape_train = (
                (shape_groups != held_out)
                & (scenario_shape != held_out_shape)
                & proposed_shape
            )
            shape_test = (shape_groups == held_out) & (scenario_shape == held_out_shape)
            if not shape_test.any():
                continue
            combined_x = np.vstack((train_x[box_train], shape_x[shape_train]))
            combined_y = np.concatenate((train_y[box_train], shape_y[shape_train]))
            combined_records = [
                record for record, keep in zip(train_records, box_train) if keep
            ] + [
                record for record, keep in zip(shape_records, shape_train) if keep
            ]
            weight = sample_weights(combined_y, combined_records)
            linear_model = fit_logistic_features(
                combined_x, combined_y, weight, linear, steps=1500, l2=0.06
            )
            tree_model = extra_trees()
            tree_model.fit(combined_x, combined_y, sample_weight=weight)
            calibration_x = train_x[box_train]
            calibration_records = [
                record for record, keep in zip(train_records, box_train) if keep
            ]
            calibration = (
                0.25 * linear_model.score(calibration_x)
                + 0.75 * tree_model.predict_proba(calibration_x)[:, 1]
            )
            test_scores = (
                0.25 * linear_model.score(shape_x[shape_test])
                + 0.75 * tree_model.predict_proba(shape_x[shape_test])[:, 1]
            )
            threshold = threshold_from_real(calibration, calibration_records)
            test_records = [
                record for record, keep in zip(shape_records, shape_test) if keep
            ]
            row = metrics(test_scores, shape_y[shape_test], test_records, threshold)
            unseen_shape_folds.append({
                "held_out_bag": held_out,
                "held_out_shape": held_out_shape,
                "threshold": threshold,
                **row,
            })
    summary["leave_shape_and_bag_out_augmented_hybrid"] = {
        "folds": unseen_shape_folds,
        "mean_positive_recall": float(np.mean([
            row["positive_recall"] for row in unseen_shape_folds
        ])),
        "worst_positive_recall": float(np.min([
            row["positive_recall"] for row in unseen_shape_folds
        ])),
        "mean_proposal_recall": float(np.mean([
            row["proposal_recall"] for row in unseen_shape_folds
        ])),
        "mean_negative_specificity": None,
    }
    report = {
        "protocol": {
            "training_shapes": "single axis-aligned ray-consistent boxes only",
            "test_shapes": "held-out composite person, L-debris, forked cable, outside stack, overhead bracket",
            "split": "leave-one-recording-out; complex shapes are test-only",
            "threshold": "training real-frame scores only",
            "augmentation_audit": "separate seen-shape bag holdout and joint shape-plus-bag holdout",
        },
        "cases": int(len(shape_y)),
        "models": summary,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--box-cache", type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--shape-cache", type=Path,
        default=Path("lidar_geometry/artifacts/complex_shape_cache_v1.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("lidar_geometry/artifacts/shape_generalization.json"),
    )
    args = parser.parse_args()
    if not args.shape_cache.exists():
        build_cache(args.dataset_root, args.shape_cache)
    evaluate(args.box_cache, args.shape_cache, args.output)


if __name__ == "__main__":
    main()
