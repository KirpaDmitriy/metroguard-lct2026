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
from lidar_geometry.risk_model import FEATURE_NAMES
from lidar_geometry.tree_ensemble_experiment import SEED, summarize


HANGING_MIN_HEIGHT_M = 0.5


def fit_hybrid(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_records: list[dict],
):
    weight = sample_weights(train_y, train_records)
    linear_model = fit_logistic_features(
        train_x, train_y, weight, linear, steps=1500, l2=0.06
    )
    tree_model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=1,
    )
    tree_model.fit(train_x, train_y, sample_weight=weight)

    def score(values: np.ndarray) -> np.ndarray:
        linear_score = linear_model.score(values)
        tree_score = tree_model.predict_proba(values)[:, 1]
        return 0.25 * linear_score + 0.75 * tree_score

    return score


def routed_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_records: list[dict],
    score_x: np.ndarray,
) -> np.ndarray:
    height_index = FEATURE_NAMES.index("height_min_m")
    train_hanging = train_x[:, height_index] >= HANGING_MIN_HEIGHT_M
    score_hanging = score_x[:, height_index] >= HANGING_MIN_HEIGHT_M
    scores = np.empty(len(score_x), dtype=np.float64)
    for route in (False, True):
        train_route = train_hanging == route
        score_route = score_hanging == route
        route_records = [
            record for record, keep in zip(train_records, train_route) if keep
        ]
        model = fit_hybrid(
            train_x[train_route], train_y[train_route], route_records
        )
        scores[score_route] = model(score_x[score_route])
    return scores


def positive_recall_by_route(
    scores: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    height_index = FEATURE_NAMES.index("height_min_m")
    hanging = features[:, height_index] >= HANGING_MIN_HEIGHT_M
    output = {}
    for name, route in (("ground", ~hanging), ("hanging", hanging)):
        positive = route & (labels == 1)
        output[name] = (
            float((scores[positive] >= threshold).mean())
            if positive.any() else None
        )
    return output


def evaluate(cache: Path, output: Path) -> dict:
    data = np.load(cache, allow_pickle=False)
    features = data["features"].astype(np.float64)
    labels = data["labels"]
    groups = data["groups"]
    records = json.loads(str(data["records_json"]))
    baseline_folds = []
    routed_folds = []
    for held_out in sorted(set(groups.tolist())):
        train = groups != held_out
        test = ~train
        train_records = [record for record, keep in zip(records, train) if keep]
        test_records = [record for record, keep in zip(records, test) if keep]

        baseline = fit_hybrid(features[train], labels[train], train_records)
        baseline_train = baseline(features[train])
        baseline_test = baseline(features[test])
        baseline_threshold = threshold_from_real(baseline_train, train_records)
        baseline_folds.append({
            "held_out_bag": held_out,
            **fold_metrics(
                baseline_test, labels[test], test_records, baseline_threshold
            ),
            "positive_recall_by_route": positive_recall_by_route(
                baseline_test, labels[test], features[test], baseline_threshold
            ),
        })

        routed_train = routed_scores(
            features[train], labels[train], train_records, features[train]
        )
        routed_test = routed_scores(
            features[train], labels[train], train_records, features[test]
        )
        routed_threshold = threshold_from_real(routed_train, train_records)
        routed_folds.append({
            "held_out_bag": held_out,
            **fold_metrics(
                routed_test, labels[test], test_records, routed_threshold
            ),
            "positive_recall_by_route": positive_recall_by_route(
                routed_test, labels[test], features[test], routed_threshold
            ),
        })

    report = {
        "protocol": {
            "split": "leave-one-organizer-bag-out",
            "change": "one fixed vertical router with independent ground and hanging rankers",
            "router": f"height_min_m >= {HANGING_MIN_HEIGHT_M}",
            "ranker": "25% linear logistic plus 75% 100-tree Extra Trees",
            "threshold": "99th percentile of training real-frame maximum scores",
            "selection": "router and model settings fixed before evaluation",
            "seed": SEED,
        },
        "baseline": summarize(baseline_folds),
        "vertical_experts": summarize(routed_folds),
    }
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
        default=Path("lidar_geometry/artifacts/vertical_expert_comparison.json"),
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cache, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
