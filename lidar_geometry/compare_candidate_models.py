from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from lidar_geometry.risk_model import FEATURE_NAMES


CRITICAL_SCENARIOS = {"minimum_on_rail", "minimum_edge_inside", "hanging_cable"}


@dataclass
class Predictor:
    score: object
    parameters: dict | None = None


def weighted_scale(x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(x, axis=0, weights=weight)
    variance = np.average((x - mean) ** 2, axis=0, weights=weight)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def sample_weights(y: np.ndarray, records: list[dict]) -> np.ndarray:
    positive = len(y) / max(1.0, 2 * y.sum())
    negative = len(y) / max(1.0, 2 * (len(y) - y.sum()))
    weight = np.where(y == 1, positive, negative)
    critical = np.asarray([
        record.get("scenario") in CRITICAL_SCENARIOS and label == 1
        for record, label in zip(records, y)
    ])
    weight[critical] *= 4
    return weight


def fit_logistic_features(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    transform,
    *,
    steps: int = 1200,
    learning_rate: float = 0.025,
    l2: float = 0.04,
) -> Predictor:
    raw_mean, raw_scale = weighted_scale(x, weight)
    base = (x - raw_mean) / raw_scale
    expanded = transform(base)
    mean, scale = weighted_scale(expanded, weight)
    z = (expanded - mean) / scale
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    normalizer = weight.sum()
    for _ in range(steps):
        logits = np.clip(z @ weights + bias, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        error = (probability - y) * weight
        weights -= learning_rate * (z.T @ error / normalizer + l2 * weights)
        bias -= learning_rate * error.sum() / normalizer

    def predict(values: np.ndarray) -> np.ndarray:
        transformed = transform((values - raw_mean) / raw_scale)
        logits = np.clip((transformed - mean) / scale @ weights + bias, -30, 30)
        return 1 / (1 + np.exp(-logits))

    return Predictor(predict, {
        "raw_mean": raw_mean,
        "raw_scale": raw_scale,
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "bias": bias,
    })


def linear(values: np.ndarray) -> np.ndarray:
    return values


def quadratic(values: np.ndarray) -> np.ndarray:
    columns = [values, values * values]
    products = [
        values[:, left] * values[:, right]
        for left in range(values.shape[1])
        for right in range(left + 1, values.shape[1])
    ]
    columns.append(np.column_stack(products))
    return np.column_stack(columns)


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    *,
    hidden: int = 8,
    steps: int = 1800,
    learning_rate: float = 0.01,
    l2: float = 0.02,
) -> Predictor:
    mean, scale = weighted_scale(x, weight)
    z = (x - mean) / scale
    rng = np.random.default_rng(20260928)
    w1 = rng.normal(0, 0.12, (z.shape[1], hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.12, hidden)
    b2 = 0.0
    parameters = [w1, b1, w2]
    first = [np.zeros_like(item) for item in parameters]
    second = [np.zeros_like(item) for item in parameters]
    first_b2 = second_b2 = 0.0
    normalizer = weight.sum()
    for step in range(1, steps + 1):
        hidden_value = np.tanh(z @ w1 + b1)
        probability = 1 / (1 + np.exp(-np.clip(hidden_value @ w2 + b2, -30, 30)))
        output_error = (probability - y) * weight / normalizer
        gradients = [
            z.T @ (output_error[:, None] * w2 * (1 - hidden_value**2)) + l2 * w1,
            (output_error[:, None] * w2 * (1 - hidden_value**2)).sum(axis=0),
            hidden_value.T @ output_error + l2 * w2,
        ]
        gradient_b2 = output_error.sum()
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first[index] = 0.9 * first[index] + 0.1 * gradient
            second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
            corrected_first = first[index] / (1 - 0.9**step)
            corrected_second = second[index] / (1 - 0.999**step)
            parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        first_b2 = 0.9 * first_b2 + 0.1 * gradient_b2
        second_b2 = 0.999 * second_b2 + 0.001 * gradient_b2 * gradient_b2
        b2 -= learning_rate * (first_b2 / (1 - 0.9**step)) / (
            np.sqrt(second_b2 / (1 - 0.999**step)) + 1e-8
        )

    def predict(values: np.ndarray) -> np.ndarray:
        hidden_value = np.tanh((values - mean) / scale @ w1 + b1)
        logits = np.clip(hidden_value @ w2 + b2, -30, 30)
        return 1 / (1 + np.exp(-logits))

    return Predictor(predict)


def fit_normal_distance(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> Predictor:
    normal = x[y == 0]
    mean = normal.mean(axis=0)
    scale = normal.std(axis=0)
    scale[scale < 1e-6] = 1.0

    def predict(values: np.ndarray) -> np.ndarray:
        distance = np.mean(((values - mean) / scale) ** 2, axis=1)
        return 1 - np.exp(-distance / 2)

    return Predictor(predict)


def threshold_from_real(scores: np.ndarray, records: list[dict]) -> float:
    maxima: dict[tuple[str, int], float] = {}
    for score, record in zip(scores, records):
        if record["source"] != "real":
            continue
        key = record["bag"], record["frame"]
        maxima[key] = max(float(score), maxima.get(key, 0.0))
    return float(np.quantile(list(maxima.values()), 0.99, method="higher"))


def fold_metrics(
    scores: np.ndarray,
    y: np.ndarray,
    records: list[dict],
    threshold: float,
) -> dict:
    predicted = scores >= threshold
    positive = y == 1
    synthetic_negative = np.asarray([
        value for value, label, record in zip(predicted, y, records)
        if label == 0 and record["source"] == "synthetic"
    ])
    real_frames: dict[tuple[str, int], bool] = {}
    for value, record in zip(predicted, records):
        if record["source"] != "real":
            continue
        key = record["bag"], record["frame"]
        real_frames[key] = bool(value) or real_frames.get(key, False)
    return {
        "threshold": threshold,
        "positive_recall": float(predicted[positive].mean()),
        "synthetic_negative_specificity": float(1 - synthetic_negative.mean()),
        "real_frame_alarm_rate": float(sum(real_frames.values()) / len(real_frames)),
        "test_candidates": len(y),
    }


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    cached = np.load(path, allow_pickle=False)
    x = cached["x"].astype(np.float64)
    if x.shape[1] == len(FEATURE_NAMES) - 1:
        lateral = x[:, FEATURE_NAMES.index("lateral_span_m")]
        center = x[:, FEATURE_NAMES.index("abs_lateral_center_m")]
        x = np.column_stack((x, 1.05 - center - lateral / 2))
    return x, cached["y"], cached["groups"], json.loads(str(cached["records_json"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/candidate_model_comparison.json"),
    )
    args = parser.parse_args()
    x, y, groups, records = load_cache(args.cache)
    fitters = {
        "linear_g4_protocol": lambda a, b, c: fit_logistic_features(a, b, c, linear),
        "quadratic_logistic": lambda a, b, c: fit_logistic_features(a, b, c, quadratic),
        "tiny_mlp_13x8x1": fit_mlp,
        "normal_distance": fit_normal_distance,
    }
    report = {
        "protocol": {
            "split": "leave-one-recording-out",
            "threshold": "99th percentile of training real-frame maximum scores",
            "critical_positive_weight": 4,
            "features": FEATURE_NAMES,
            "seed": 20260928,
        },
        "models": {},
    }
    for name, fitter in fitters.items():
        folds = []
        for held_out in sorted(set(groups.tolist())):
            train = groups != held_out
            test = ~train
            train_records = [record for record, keep in zip(records, train) if keep]
            test_records = [record for record, keep in zip(records, test) if keep]
            weight = sample_weights(y[train], train_records)
            predictor = fitter(x[train], y[train], weight)
            threshold = threshold_from_real(predictor.score(x[train]), train_records)
            folds.append({
                "held_out_bag": held_out,
                **fold_metrics(predictor.score(x[test]), y[test], test_records, threshold),
            })
        report["models"][name] = {
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["models"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
