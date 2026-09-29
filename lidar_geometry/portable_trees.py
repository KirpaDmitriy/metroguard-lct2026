from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PortableExtraTrees:
    offsets: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    positive_probability: np.ndarray

    @classmethod
    def from_sklearn(cls, model) -> "PortableExtraTrees":
        offsets = [0]
        feature = []
        threshold = []
        left = []
        right = []
        probability = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            base = offsets[-1]
            values = tree.value[:, 0, :]
            normalizer = values.sum(axis=1)
            feature.extend(tree.feature.tolist())
            threshold.extend(tree.threshold.tolist())
            left.extend(
                np.where(tree.children_left >= 0, tree.children_left + base, -1)
            )
            right.extend(
                np.where(tree.children_right >= 0, tree.children_right + base, -1)
            )
            probability.extend(
                np.divide(
                    values[:, 1],
                    normalizer,
                    out=np.zeros(len(values)),
                    where=normalizer > 0,
                )
            )
            offsets.append(base + tree.node_count)
        return cls(
            np.asarray(offsets, dtype=np.int32),
            np.asarray(feature, dtype=np.int16),
            np.asarray(threshold, dtype=np.float64),
            np.asarray(left, dtype=np.int32),
            np.asarray(right, dtype=np.int32),
            np.asarray(probability, dtype=np.float64),
        )

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return np.empty(0, dtype=np.float64)
        roots = self.offsets[:-1]
        nodes = np.broadcast_to(roots, (len(values), len(roots))).copy()
        while True:
            node_features = self.feature[nodes]
            active_rows, active_trees = np.nonzero(node_features >= 0)
            if not len(active_rows):
                break
            active_nodes = nodes[active_rows, active_trees]
            features = self.feature[active_nodes]
            go_left = values[active_rows, features] <= self.threshold[active_nodes]
            nodes[active_rows, active_trees] = np.where(
                go_left,
                self.left[active_nodes],
                self.right[active_nodes],
            )
        return self.positive_probability[nodes].mean(axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            offsets=self.offsets,
            feature=self.feature,
            threshold=self.threshold,
            left=self.left,
            right=self.right,
            positive_probability=self.positive_probability,
        )

    @classmethod
    def load(cls, path: Path) -> "PortableExtraTrees":
        data = np.load(path, allow_pickle=False)
        return cls(
            *(
                data[name]
                for name in (
                    "offsets",
                    "feature",
                    "threshold",
                    "left",
                    "right",
                    "positive_probability",
                )
            )
        )


@dataclass(frozen=True)
class PortableFarRangeRanker:
    tree: PortableExtraTrees
    threshold: float

    def score(self, values: np.ndarray) -> np.ndarray:
        return self.tree.score(values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            offsets=self.tree.offsets,
            feature=self.tree.feature,
            tree_threshold=self.tree.threshold,
            left=self.tree.left,
            right=self.tree.right,
            positive_probability=self.tree.positive_probability,
            score_threshold=np.asarray(self.threshold),
        )

    @classmethod
    def load(cls, path: Path) -> "PortableFarRangeRanker":
        data = np.load(path, allow_pickle=False)
        tree = PortableExtraTrees(
            data["offsets"],
            data["feature"],
            data["tree_threshold"],
            data["left"],
            data["right"],
            data["positive_probability"],
        )
        return cls(tree, float(data["score_threshold"]))


@dataclass(frozen=True)
class PortableLinearTreeHybrid:
    tree: PortableExtraTrees
    raw_mean: np.ndarray
    raw_scale: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    tree_weight: float
    threshold: float

    def score(self, values: np.ndarray) -> np.ndarray:
        base = (values - self.raw_mean) / self.raw_scale
        logits = np.clip(
            (base - self.mean) / self.scale @ self.weights + self.bias, -30, 30
        )
        linear_score = 1 / (1 + np.exp(-logits))
        return (
            1 - self.tree_weight
        ) * linear_score + self.tree_weight * self.tree.score(values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            offsets=self.tree.offsets,
            feature=self.tree.feature,
            tree_threshold=self.tree.threshold,
            left=self.tree.left,
            right=self.tree.right,
            positive_probability=self.tree.positive_probability,
            raw_mean=self.raw_mean,
            raw_scale=self.raw_scale,
            linear_mean=self.mean,
            linear_scale=self.scale,
            weights=self.weights,
            bias=np.asarray(self.bias),
            tree_weight=np.asarray(self.tree_weight),
            decision_threshold=np.asarray(self.threshold),
        )

    @classmethod
    def load(cls, path: Path) -> "PortableLinearTreeHybrid":
        data = np.load(path, allow_pickle=False)
        tree = PortableExtraTrees(
            data["offsets"],
            data["feature"],
            data["tree_threshold"],
            data["left"],
            data["right"],
            data["positive_probability"],
        )
        return cls(
            tree,
            data["raw_mean"],
            data["raw_scale"],
            data["linear_mean"],
            data["linear_scale"],
            data["weights"],
            float(data["bias"]),
            float(data["tree_weight"]),
            float(data["decision_threshold"]),
        )
