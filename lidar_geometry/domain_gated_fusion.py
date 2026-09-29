from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    fold_metrics,
    linear,
    sample_weights,
    threshold_from_real,
)
from lidar_geometry.domain_guard import context_descriptor
from lidar_geometry.spatial_patch_experiment import convolution_descriptor


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def nearest_distance(reference: np.ndarray, query: np.ndarray, leave_self_out: bool) -> np.ndarray:
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 256):
        block = query[start:start + 256]
        distance = ((block[:, None, :] - reference[None, :, :]) ** 2).mean(axis=2)
        if leave_self_out:
            rows = np.arange(len(block))
            distance[rows, start + rows] = np.inf
        result[start:start + len(block)] = distance.min(axis=1)
    return result


def domain_confidence(
    context: np.ndarray,
    train_real: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    reference = context[train_real]
    median = np.median(reference, axis=0)
    scale = np.median(np.abs(reference - median), axis=0) * 1.4826
    scale[scale < 1e-4] = 1.0
    reference = np.clip((reference - median) / scale, -8, 8)
    train_values = np.clip((context[train] - median) / scale, -8, 8)
    test_values = np.clip((context[test] - median) / scale, -8, 8)

    reference_positions = {index: position for position, index in enumerate(np.flatnonzero(train_real))}
    train_distance = nearest_distance(reference, train_values, False)
    for local, index in enumerate(np.flatnonzero(train)):
        if index not in reference_positions:
            continue
        delta = ((train_values[local] - reference) ** 2).mean(axis=1)
        delta[reference_positions[index]] = np.inf
        train_distance[local] = delta.min()
    reference_loo = nearest_distance(reference, reference, True)
    test_distance = nearest_distance(reference, test_values, False)
    radius = max(float(np.quantile(reference_loo, 0.95)), 1e-6)

    def confidence(distance: np.ndarray) -> np.ndarray:
        return np.exp(-np.log(2) * distance / radius)

    return confidence(train_distance), confidence(test_distance), {
        "reference_candidates": len(reference),
        "in_domain_radius": radius,
        "test_median_distance_ratio": float(np.median(test_distance) / radius),
        "test_mean_cv_weight_multiplier": float(confidence(test_distance).mean()),
    }


def summarize(folds: list[dict]) -> dict:
    return {
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


def evaluate(cache: Path, output: Path) -> None:
    data = np.load(cache, allow_pickle=False)
    component = data["features"].astype(np.float64)
    patches = data["patches"]
    spatial = convolution_descriptor(patches).astype(np.float64)
    context = context_descriptor(patches)
    labels = data["labels"]
    groups = data["groups"]
    records = json.loads(str(data["records_json"]))
    names = sorted(set(groups.tolist()))
    strategies = {
        "geometry": (0.0, False),
        "static_25": (0.25, False),
        "static_50": (0.50, False),
        "static_75": (0.75, False),
        "domain_gated_25": (0.25, True),
        "domain_gated_50": (0.50, True),
        "domain_gated_75": (0.75, True),
        "domain_gated_100": (1.00, True),
        "cv": (1.0, False),
    }
    results = {name: [] for name in strategies}
    abstention_results = {f"abstain_below_{level}": [] for level in (0.25, 0.50, 0.75)}
    abstention_results["abstain_below_0.25_with_strong_bypass"] = []
    for margin in (0.25, 0.50, 0.75, 1.25):
        abstention_results[f"abstain_0.25_bypass_margin_{margin}"] = []
    domain_audit = []

    for held_out in names:
        train = groups != held_out
        test = ~train
        train_records = [record for record, keep in zip(records, train) if keep]
        test_records = [record for record, keep in zip(records, test) if keep]
        weight = sample_weights(labels[train], train_records)
        geometry = fit_logistic_features(component[train], labels[train], weight, linear, steps=1500, l2=0.06)
        vision = fit_logistic_features(spatial[train], labels[train], weight, linear, steps=1500, l2=0.06)
        geometry_train = geometry.score(component[train])
        geometry_test = geometry.score(component[test])
        vision_train = vision.score(spatial[train])
        vision_test = vision.score(spatial[test])
        geometry_threshold = threshold_from_real(geometry_train, train_records)
        vision_threshold = threshold_from_real(vision_train, train_records)
        geometry_margin_train = logit(geometry_train) - logit(np.asarray(geometry_threshold))
        geometry_margin_test = logit(geometry_test) - logit(np.asarray(geometry_threshold))
        vision_margin_train = logit(vision_train) - logit(np.asarray(vision_threshold))
        vision_margin_test = logit(vision_test) - logit(np.asarray(vision_threshold))

        real = np.asarray([record["source"] == "real" for record in records])
        confidence_train, confidence_test, audit = domain_confidence(
            context, train & real, train, test
        )
        domain_audit.append({"held_out_bag": held_out, **audit})

        for name, (maximum_weight, gated) in strategies.items():
            train_weight = maximum_weight * (confidence_train if gated else 1.0)
            test_weight = maximum_weight * (confidence_test if gated else 1.0)
            score_train = geometry_margin_train + train_weight * (
                vision_margin_train - geometry_margin_train
            )
            score_test = geometry_margin_test + test_weight * (
                vision_margin_test - geometry_margin_test
            )
            threshold = threshold_from_real(score_train, train_records)
            results[name].append({
                "held_out_bag": held_out,
                **fold_metrics(score_test, labels[test], test_records, threshold),
            })
        for level in (0.25, 0.50, 0.75):
            abstained_score = np.where(confidence_test >= level, geometry_margin_test, -np.inf)
            abstention_results[f"abstain_below_{level}"].append({
                "held_out_bag": held_out,
                "abstained_candidate_rate": float((confidence_test < level).mean()),
                **fold_metrics(abstained_score, labels[test], test_records, 0.0),
            })
        strong_geometry = geometry_margin_test >= 1.0
        bypass_score = np.where(
            (confidence_test >= 0.25) | strong_geometry,
            geometry_margin_test,
            -np.inf,
        )
        abstention_results["abstain_below_0.25_with_strong_bypass"].append({
            "held_out_bag": held_out,
            "abstained_candidate_rate": float(
                ((confidence_test < 0.25) & ~strong_geometry).mean()
            ),
            **fold_metrics(bypass_score, labels[test], test_records, 0.0),
        })
        for margin in (0.25, 0.50, 0.75, 1.25):
            strong_geometry = geometry_margin_test >= margin
            bypass_score = np.where(
                (confidence_test >= 0.25) | strong_geometry,
                geometry_margin_test,
                -np.inf,
            )
            abstention_results[f"abstain_0.25_bypass_margin_{margin}"].append({
                "held_out_bag": held_out,
                "abstained_candidate_rate": float(
                    ((confidence_test < 0.25) & ~strong_geometry).mean()
                ),
                **fold_metrics(bypass_score, labels[test], test_records, 0.0),
            })

    report = {
        "protocol": {
            "split": "leave-one-recording-out",
            "domain_reference": "training real candidates only",
            "domain_signature": "24 robust outer-patch and quadrant statistics",
            "fusion": "threshold-relative logit margins",
            "sweep": "fixed in advance; no selection on held-out folds",
            "threshold": "99th percentile of training real-frame maximum fused scores",
        },
        "samples": len(labels),
        "domain_audit": domain_audit,
        "strategies": {name: summarize(folds) for name, folds in results.items()},
        "abstention": {name: summarize(folds) for name, folds in abstention_results.items()},
    }
    geometry_folds = report["strategies"]["geometry"]["folds"]
    cv_folds = report["strategies"]["cv"]["folds"]
    report["oracle_audit"] = {
        "folds_where_cv_recall_beats_geometry": sum(
            cv["positive_recall"] > geometry["positive_recall"]
            for geometry, cv in zip(geometry_folds, cv_folds)
        ),
        "folds_where_cv_alarm_rate_beats_geometry": sum(
            cv["real_frame_alarm_rate"] < geometry["real_frame_alarm_rate"]
            for geometry, cv in zip(geometry_folds, cv_folds)
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/domain_gated_fusion.json"),
    )
    args = parser.parse_args()
    evaluate(args.cache, args.output)


if __name__ == "__main__":
    main()
