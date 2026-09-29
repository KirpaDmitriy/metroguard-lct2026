from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    fold_metrics,
    linear,
    sample_weights,
    threshold_from_real,
)


SEED = 20260929


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


def evaluate(cache: Path, output: Path) -> dict:
    data = np.load(cache, allow_pickle=False)
    features = data["features"].astype(np.float64)
    labels = data["labels"]
    groups = data["groups"]
    records = json.loads(str(data["records_json"]))
    factories = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=SEED,
        ),
    }
    report = {
        "protocol": {
            "split": "leave-one-recording-out",
            "threshold": "99th percentile of training real-frame maximum scores",
            "selection": "fixed hyperparameters; no held-out fold tuning",
            "features": int(features.shape[1]),
            "samples": int(len(labels)),
            "seed": SEED,
            "scikit_learn": sklearn.__version__,
        },
        "models": {},
    }
    held_out_groups = sorted(set(groups.tolist()))
    for name in ("component_linear", *factories):
        folds = []
        for held_out in held_out_groups:
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            test_records = [record for record, keep in zip(records, test) if keep]
            weight = sample_weights(labels[train], train_records)
            if name == "component_linear":
                predictor = fit_logistic_features(
                    features[train], labels[train], weight, linear, steps=1500, l2=0.06
                )
                train_scores = predictor.score(features[train])
                test_scores = predictor.score(features[test])
            else:
                model = factories[name]()
                model.fit(features[train], labels[train], sample_weight=weight)
                train_scores = model.predict_proba(features[train])[:, 1]
                test_scores = model.predict_proba(features[test])[:, 1]
            threshold = threshold_from_real(train_scores, train_records)
            folds.append({
                "held_out_bag": held_out,
                **fold_metrics(test_scores, labels[test], test_records, threshold),
            })
        report["models"][name] = summarize(folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


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
        default=Path("lidar_geometry/artifacts/tree_ensemble_comparison.json"),
    )
    args = parser.parse_args()
    report = evaluate(args.cache, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
