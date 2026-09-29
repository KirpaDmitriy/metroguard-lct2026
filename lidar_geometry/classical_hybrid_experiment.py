from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    fold_metrics,
    linear,
    sample_weights,
    threshold_from_real,
)
from lidar_geometry.tree_ensemble_experiment import SEED, summarize


def extra_trees(n_estimators: int = 300) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=1,
    )


def fit_base_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weight: np.ndarray,
    *score_x: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    linear_model = fit_logistic_features(
        train_x, train_y, weight, linear, steps=1500, l2=0.06
    )
    tree_model = extra_trees()
    tree_model.fit(train_x, train_y, sample_weight=weight)
    linear_scores = [linear_model.score(values) for values in score_x]
    tree_scores = [tree_model.predict_proba(values)[:, 1] for values in score_x]
    return linear_scores, tree_scores


def standalone_scores(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    weight: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if name == "rbf_svm":
        model = make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", kernel="rbf"))
        model.fit(train_x, train_y, svc__sample_weight=weight)
        return model.decision_function(train_x), model.decision_function(test_x)
    if name == "knn_distance_15":
        model = make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance")
        )
        model.fit(train_x, train_y)
    elif name == "gaussian_naive_bayes":
        model = GaussianNB(var_smoothing=1e-8)
        model.fit(train_x, train_y, sample_weight=weight)
    elif name == "adaboost_stumps":
        model = AdaBoostClassifier(
            n_estimators=200, learning_rate=0.05, random_state=SEED
        )
        model.fit(train_x, train_y, sample_weight=weight)
    else:
        raise ValueError(name)
    return model.predict_proba(train_x)[:, 1], model.predict_proba(test_x)[:, 1]


def stacked_scores(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    records: list[dict],
    outer_train: np.ndarray,
    outer_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    train_indices = np.flatnonzero(outer_train)
    oof = np.zeros((len(train_indices), 2), dtype=np.float64)
    outer_groups = groups[outer_train]
    outer_records = [record for record, keep in zip(records, outer_train) if keep]
    for inner_group in sorted(set(outer_groups.tolist())):
        inner_train = outer_groups != inner_group
        inner_valid = ~inner_train
        inner_records = [
            record for record, keep in zip(outer_records, inner_train) if keep
        ]
        weight = sample_weights(labels[train_indices][inner_train], inner_records)
        linear_scores, tree_scores = fit_base_scores(
            features[train_indices][inner_train],
            labels[train_indices][inner_train],
            weight,
            features[train_indices][inner_valid],
        )
        oof[inner_valid, 0] = linear_scores[0]
        oof[inner_valid, 1] = tree_scores[0]
    outer_weight = sample_weights(labels[outer_train], outer_records)
    meta = fit_logistic_features(
        oof, labels[outer_train], outer_weight, linear, steps=1000, l2=0.1
    )
    linear_scores, tree_scores = fit_base_scores(
        features[outer_train], labels[outer_train], outer_weight,
        features[outer_train], features[outer_test],
    )
    test_base = np.column_stack((linear_scores[1], tree_scores[1]))
    return meta.score(oof), meta.score(test_base), outer_records


def evaluate(cache: Path, output: Path) -> dict:
    data = np.load(cache, allow_pickle=False)
    features = data["features"].astype(np.float64)
    labels = data["labels"]
    groups = data["groups"]
    records = json.loads(str(data["records_json"]))
    names = (
        "rbf_svm",
        "knn_distance_15",
        "gaussian_naive_bayes",
        "adaboost_stumps",
        "linear_tree_average",
        "tree_weighted_75",
        "linear_tree_max_vote",
        "linear_tree_stack",
    )
    folds = {name: [] for name in names}
    for held_out in sorted(set(groups.tolist())):
        train = groups != held_out
        test = ~train
        train_records = [record for record, keep in zip(records, train) if keep]
        test_records = [record for record, keep in zip(records, test) if keep]
        weight = sample_weights(labels[train], train_records)
        linear_scores, tree_scores = fit_base_scores(
            features[train], labels[train], weight, features[train], features[test]
        )
        hybrids = {
            "linear_tree_average": (
                0.5 * linear_scores[0] + 0.5 * tree_scores[0],
                0.5 * linear_scores[1] + 0.5 * tree_scores[1],
            ),
            "tree_weighted_75": (
                0.25 * linear_scores[0] + 0.75 * tree_scores[0],
                0.25 * linear_scores[1] + 0.75 * tree_scores[1],
            ),
            "linear_tree_max_vote": (
                np.maximum(linear_scores[0], tree_scores[0]),
                np.maximum(linear_scores[1], tree_scores[1]),
            ),
        }
        for name in names[:4]:
            train_score, test_score = standalone_scores(
                name, features[train], labels[train], weight, features[test]
            )
            threshold = threshold_from_real(train_score, train_records)
            folds[name].append({
                "held_out_bag": held_out,
                **fold_metrics(test_score, labels[test], test_records, threshold),
            })
        for name, (train_score, test_score) in hybrids.items():
            threshold = threshold_from_real(train_score, train_records)
            folds[name].append({
                "held_out_bag": held_out,
                **fold_metrics(test_score, labels[test], test_records, threshold),
            })
        stack_train, stack_test, stack_records = stacked_scores(
            features, labels, groups, records, train, test
        )
        threshold = threshold_from_real(stack_train, stack_records)
        folds["linear_tree_stack"].append({
            "held_out_bag": held_out,
            **fold_metrics(stack_test, labels[test], test_records, threshold),
        })
        print(f"evaluated held-out bag: {held_out}", flush=True)
    report = {
        "protocol": {
            "split": "leave-one-recording-out",
            "threshold": "99th percentile of training real-frame maximum scores",
            "selection": "fixed hyperparameters and fixed hybrid weights",
            "stacking": "inner leave-one-recording-out predictions for meta training",
            "samples": int(len(labels)),
            "seed": SEED,
        },
        "models": {name: summarize(value) for name, value in folds.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("lidar_geometry/artifacts/classical_hybrid_comparison.json"),
    )
    args = parser.parse_args()
    report = evaluate(args.cache, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
