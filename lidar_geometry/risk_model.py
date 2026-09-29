"""Tiny, dependency-free proposal ranker for the hybrid MVP.

Geometry remains responsible for safety-volume construction and candidate
generation.  This model only answers the narrower question exposed by the
baseline experiment: does a geometric component look more like normal track
infrastructure or like a compact inserted object?

The serialized model is just feature normalization plus logistic-regression
weights, so inference is deterministic, inspectable and cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from lidar_geometry.detect_obstacles import Obstacle


FEATURE_NAMES = (
    "log_points",
    "log_voxels",
    "forward_span_m",
    "lateral_span_m",
    "height_span_m",
    "height_min_m",
    "height_max_m",
    "residual_mean_m",
    "residual_max_m",
    "abs_lateral_center_m",
    "log_density",
    "range_scaled_points",
    "side_clearance_m",
)


def feature_values(item: Obstacle) -> dict[str, float]:
    forward = max(0.01, item.distance_max_m - item.distance_min_m)
    lateral = max(0.01, item.lateral_max_m - item.lateral_min_m)
    height = max(0.01, item.height_max_m - item.height_min_m)
    volume = forward * lateral * height
    distance = max(3.0, item.distance_min_m)
    lateral_center = abs((item.lateral_min_m + item.lateral_max_m) / 2)
    return {
        "log_points": math.log1p(item.points),
        "log_voxels": math.log1p(item.voxels),
        "forward_span_m": forward,
        "lateral_span_m": lateral,
        "height_span_m": height,
        "height_min_m": item.height_min_m,
        "height_max_m": item.height_max_m,
        "residual_mean_m": item.surface_residual_mean_m,
        "residual_max_m": item.surface_residual_max_m,
        "abs_lateral_center_m": lateral_center,
        "log_density": math.log1p(item.points / volume),
        "range_scaled_points": math.log1p(item.points * (distance / 20.0) ** 2),
        "side_clearance_m": 1.05 - lateral_center - lateral / 2,
    }


def component_features(
    item: Obstacle,
    names: tuple[str, ...] = FEATURE_NAMES,
) -> np.ndarray:
    values = feature_values(item)
    return np.asarray([values[name] for name in names], dtype=np.float64)


@dataclass(frozen=True)
class RiskModel:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    threshold: float

    def score(self, item: Obstacle) -> float:
        x = component_features(item, self.feature_names)
        z = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        logit = float(z @ np.asarray(self.weights) + self.bias)
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format": "metro_guard_logistic_v1",
            "feature_names": self.feature_names,
            "mean": self.mean,
            "scale": self.scale,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RiskModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        feature_names = tuple(payload["feature_names"])
        unknown = set(feature_names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"risk-model contains unknown features: {sorted(unknown)}")
        return cls(
            feature_names, tuple(payload["mean"]), tuple(payload["scale"]),
            tuple(payload["weights"]), float(payload["bias"]),
            float(payload["threshold"]),
        )


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int = 2500,
    learning_rate: float = 0.025,
    l2: float = 0.04,
    threshold: float = 0.68,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> RiskModel:
    """Fit balanced logistic regression using deterministic full-batch GD."""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = (features - mean) / scale
    y = labels.astype(np.float64)
    positive_weight = len(y) / max(1.0, 2 * y.sum())
    negative_weight = len(y) / max(1.0, 2 * (len(y) - y.sum()))
    sample_weight = np.where(y > 0.5, positive_weight, negative_weight)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        logits = np.clip(x @ weights + bias, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = (probability - y) * sample_weight
        weights -= learning_rate * (x.T @ error / len(y) + l2 * weights)
        bias -= learning_rate * error.mean()
    return RiskModel(
        feature_names, tuple(mean.tolist()), tuple(scale.tolist()), tuple(weights.tolist()),
        float(bias), threshold,
    )
