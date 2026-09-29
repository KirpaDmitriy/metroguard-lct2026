from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.domain_guard import (
    DomainGuard,
    component_patch,
)
from lidar_geometry.evidence_memory import RangeEvidenceMemory
from lidar_geometry.fast_detector import SafetyDetection, detect_fast
from lidar_geometry.hybrid_detector import detect_hybrid
from lidar_geometry.pointcloud2 import PointCloud2
from lidar_geometry.portable_trees import PortableLinearTreeHybrid
from lidar_geometry.risk_model import RiskModel, component_features

ALGORITHMS = ("geometry", "linear_hybrid", "tree_hybrid", "memory_hybrid")


def portable_tree_scores(
    geometric: SafetyDetection,
    model: PortableLinearTreeHybrid,
) -> np.ndarray:
    if not geometric.obstacles:
        return np.empty(0)
    features = np.vstack([component_features(item) for item in geometric.obstacles])
    return model.score(features)


def filter_portable_tree_detection(
    geometric: SafetyDetection,
    model: PortableLinearTreeHybrid,
    scores: np.ndarray | None = None,
) -> SafetyDetection:
    if not geometric.obstacles:
        return geometric
    if scores is None:
        scores = portable_tree_scores(geometric, model)
    accepted = [
        item
        for item, score in zip(geometric.obstacles, scores)
        if score >= model.threshold
    ]
    if accepted:
        nearest = min(accepted, key=lambda item: item.distance_min_m)
        confidence = max(float(score) for score in scores if score >= model.threshold)
        return replace(
            geometric,
            state="OBSTACLE",
            obstacle=True,
            nearest_distance_m=nearest.distance_min_m,
            confidence=confidence,
            obstacles=tuple(accepted),
            reason="geometry_plus_portable_tree_ranker",
        )
    return replace(
        geometric,
        state="UNKNOWN",
        obstacle=False,
        confidence=float(scores.max()),
        obstacles=(),
        reason="geometric_components_rejected_by_portable_tree_ranker",
    )


class RuntimeDetector:
    def __init__(
        self,
        algorithm: str,
        *,
        config: DetectorConfig = DetectorConfig(),
        risk_model_path: Path | None = None,
        portable_model_path: Path | None = None,
        domain_guard_path: Path | None = None,
        risk_threshold: float | None = None,
    ):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.algorithm = algorithm
        self.config = config
        self.linear: RiskModel | None = None
        self.tree: PortableLinearTreeHybrid | None = None
        self.memory: RangeEvidenceMemory | None = None
        self.domain_guard: DomainGuard | None = None
        self.tunnel_familiarity: float | None = None
        self.tunnel_memory_frame = 0

        if algorithm == "linear_hybrid":
            if risk_model_path is None:
                raise ValueError("linear_hybrid requires risk_model_path")
            self.linear = RiskModel.load(risk_model_path)
            if risk_threshold is not None and risk_threshold >= 0:
                self.linear = replace(self.linear, threshold=risk_threshold)
            if domain_guard_path is not None:
                self.domain_guard = DomainGuard.load(domain_guard_path)
        elif algorithm in {"tree_hybrid", "memory_hybrid"}:
            if portable_model_path is None:
                raise ValueError("tree_hybrid requires portable_model_path")
            self.tree = PortableLinearTreeHybrid.load(portable_model_path)
            if algorithm == "memory_hybrid":
                self.memory = RangeEvidenceMemory(self.tree.threshold)
                if domain_guard_path is not None and domain_guard_path.is_file():
                    self.domain_guard = DomainGuard.load(domain_guard_path)

    @classmethod
    def from_root(
        cls,
        algorithm: str,
        root: Path,
        *,
        config: DetectorConfig = DetectorConfig(),
    ) -> "RuntimeDetector":
        return cls(
            algorithm,
            config=config,
            risk_model_path=root / "lidar_geometry/models/risk_model.json",
            portable_model_path=(
                root / "lidar_geometry/artifacts/linear_extra_trees_hybrid.npz"
            ),
            domain_guard_path=root / "lidar_geometry/models/domain_guard.npz",
        )

    def __call__(self, cloud: PointCloud2) -> SafetyDetection:
        if self.linear is not None:
            return detect_hybrid(cloud, self.linear, self.config, self.domain_guard)
        context = [] if self.memory is not None else None
        geometric = detect_fast(cloud, self.config, context_out=context)
        return self.apply_geometric(
            geometric,
            context[0] if context else None,
        )

    def apply_geometric(
        self,
        geometric: SafetyDetection,
        context: tuple[np.ndarray, ...] | None = None,
    ) -> SafetyDetection:
        if self.tree is None:
            return geometric
        scores = portable_tree_scores(geometric, self.tree)
        if self.memory is not None:
            scores = self.adapt_scores_to_tunnel_memory(context, geometric, scores)
            return self.memory.update(geometric, scores)
        return filter_portable_tree_detection(geometric, self.tree, scores)

    def adapt_scores_to_tunnel_memory(
        self,
        context: tuple[np.ndarray, ...] | None,
        geometric: SafetyDetection,
        scores: np.ndarray,
    ) -> np.ndarray:
        if context is None or self.domain_guard is None or not len(scores):
            return scores
        self.tunnel_memory_frame += 1
        if self.tunnel_familiarity is None or self.tunnel_memory_frame % 50 == 0:
            anchor = min(
                geometric.obstacles,
                key=lambda item: item.distance_min_m,
            )
            patch = component_patch(context, anchor)[None, ...]
            self.tunnel_familiarity = float(self.domain_guard.confidence(patch)[0])
        familiarity = self.tunnel_familiarity
        return np.where(
            familiarity >= self.domain_guard.confidence_threshold,
            scores,
            np.maximum(scores, self.tree.threshold),
        )
