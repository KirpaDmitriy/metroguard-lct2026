from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    fold_metrics,
    linear,
    sample_weights,
    threshold_from_real,
)
from lidar_geometry.tree_ensemble_experiment import SEED, summarize


def evaluate(cache: Path, output: Path) -> dict:
    data = np.load(cache, allow_pickle=False)
    features = data["features"].astype(np.float64)
    labels = data["labels"]
    groups = data["groups"]
    records = json.loads(str(data["records_json"]))
    counts = (25, 50, 100, 150)
    folds = {
        f"extra_trees_{count}": [] for count in counts
    } | {
        f"tree_weighted_75_{count}": [] for count in counts
    }
    for held_out in sorted(set(groups.tolist())):
        train = groups != held_out
        test = ~train
        train_records = [record for record, keep in zip(records, train) if keep]
        test_records = [record for record, keep in zip(records, test) if keep]
        weight = sample_weights(labels[train], train_records)
        linear_model = fit_logistic_features(
            features[train], labels[train], weight, linear, steps=1500, l2=0.06
        )
        linear_train = linear_model.score(features[train])
        linear_test = linear_model.score(features[test])
        for count in counts:
            model = ExtraTreesClassifier(
                n_estimators=count,
                max_depth=6,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=1,
            )
            model.fit(features[train], labels[train], sample_weight=weight)
            tree_train = model.predict_proba(features[train])[:, 1]
            tree_test = model.predict_proba(features[test])[:, 1]
            score_sets = {
                f"extra_trees_{count}": (tree_train, tree_test),
                f"tree_weighted_75_{count}": (
                    0.25 * linear_train + 0.75 * tree_train,
                    0.25 * linear_test + 0.75 * tree_test,
                ),
            }
            for name, (train_score, test_score) in score_sets.items():
                threshold = threshold_from_real(train_score, train_records)
                folds[name].append({
                    "held_out_bag": held_out,
                    **fold_metrics(test_score, labels[test], test_records, threshold),
                })
    report = {
        "protocol": {
            "split": "leave-one-recording-out",
            "change": "number of Extra Trees only",
            "fixed_depth": 6,
            "fixed_min_samples_leaf": 5,
            "threshold": "99th percentile of training real-frame maximum scores",
            "seed": SEED,
        },
        "models": {name: summarize(value) for name, value in folds.items()},
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("lidar_geometry/artifacts/compact_tree_comparison.json"),
    )
    args = parser.parse_args()
    evaluate(args.cache, args.output)


if __name__ == "__main__":
    main()
