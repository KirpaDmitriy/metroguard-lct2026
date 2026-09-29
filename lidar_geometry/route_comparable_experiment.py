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


def evaluate(candidate_cache: Path, route_cache: Path, route_report: Path, output: Path) -> dict:
    candidate = np.load(candidate_cache, allow_pickle=False)
    route = np.load(route_cache, allow_pickle=False)
    candidate_x = candidate["features"].astype(np.float64)
    candidate_y = candidate["labels"]
    candidate_groups = candidate["groups"]
    candidate_records = json.loads(str(candidate["records_json"]))
    route_x = route["current_features"].astype(np.float64)
    route_y = route["labels"]
    route_groups = route["groups"]
    route_records = json.loads(str(route["records_json"]))
    methods = {name: [] for name in ("component_linear", "extra_trees_100", "tree_weighted_75_100")}
    for held_out in sorted(set(candidate_groups.tolist())):
        train = candidate_groups != held_out
        test = route_groups == held_out
        train_records = [
            record for record, keep in zip(candidate_records, train) if keep
        ]
        test_records = [
            {**record, "source": "synthetic"}
            for record, keep in zip(route_records, test) if keep
        ]
        weight = sample_weights(candidate_y[train], train_records)
        linear_model = fit_logistic_features(
            candidate_x[train], candidate_y[train], weight, linear,
            steps=1500, l2=0.06,
        )
        tree_model = extra_trees(100)
        tree_model.fit(candidate_x[train], candidate_y[train], sample_weight=weight)
        linear_train = linear_model.score(candidate_x[train])
        tree_train = tree_model.predict_proba(candidate_x[train])[:, 1]
        linear_test = linear_model.score(route_x[test])
        tree_test = tree_model.predict_proba(route_x[test])[:, 1]
        scores = {
            "component_linear": (linear_train, linear_test),
            "extra_trees_100": (tree_train, tree_test),
            "tree_weighted_75_100": (
                0.25 * linear_train + 0.75 * tree_train,
                0.25 * linear_test + 0.75 * tree_test,
            ),
        }
        for name, (calibration, test_scores) in scores.items():
            threshold = threshold_from_real(calibration, train_records)
            predicted = test_scores >= threshold
            positive = route_y[test] == 1
            negative = ~positive
            methods[name].append({
                "held_out_bag": held_out,
                "threshold": threshold,
                "positive_recall": float(predicted[positive].mean()),
                "synthetic_negative_specificity": float((~predicted[negative]).mean()),
                "test_candidates": int(test.sum()),
            })
    summaries = {}
    for name, folds in methods.items():
        summaries[name] = {
            "folds": folds,
            "mean_positive_recall": float(np.mean([row["positive_recall"] for row in folds])),
            "worst_positive_recall": float(np.min([row["positive_recall"] for row in folds])),
            "mean_negative_specificity": float(np.mean([
                row["synthetic_negative_specificity"] for row in folds
            ])),
        }
    memory = json.loads(route_report.read_text())["methods"]["clearance_coherent_max_clean"]
    report = {
        "protocol": {
            "obstacle_pairs": "identical route-memory current clouds for every method",
            "split": "leave-one-recording-out",
            "universal_training": "box candidates from training bags only",
            "route_memory_input": "clean same-location reference plus current cloud",
            "comparable_metrics": ["positive_recall", "negative_specificity"],
            "noncomparable_false_alarm_metrics": {
                "universal": "whole real held-out frames in candidate benchmark",
                "route_memory": "synthetically perturbed clean same-location pairs",
            },
        },
        "methods": {
            **summaries,
            "aligned_route_memory": {
                "mean_positive_recall": memory["mean_positive_recall"],
                "worst_positive_recall": memory["worst_positive_recall"],
                "mean_negative_specificity": memory["mean_negative_specificity"],
                "mean_clean_alarm_rate": memory["mean_clean_alarm_rate"],
                "requires_same_location_reference": True,
            },
        },
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-cache", type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--route-cache", type=Path,
        default=Path("lidar_geometry/artifacts/route_memory_pairs_v2.npz"),
    )
    parser.add_argument(
        "--route-report", type=Path,
        default=Path("lidar_geometry/artifacts/route_memory_evaluation_v2.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("lidar_geometry/artifacts/route_comparable_comparison.json"),
    )
    args = parser.parse_args()
    evaluate(args.candidate_cache, args.route_cache, args.route_report, args.output)


if __name__ == "__main__":
    main()
