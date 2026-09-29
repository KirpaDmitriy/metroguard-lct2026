"""Hybrid safety decision: geometric proposals plus a tiny risk ranker."""

from __future__ import annotations

from dataclasses import replace

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.domain_guard import DomainGuard, apply_domain_guard
from lidar_geometry.fast_detector import SafetyDetection, detect_fast
from lidar_geometry.pointcloud2 import PointCloud2
from lidar_geometry.risk_model import RiskModel


def detect_hybrid(
    cloud: PointCloud2,
    model: RiskModel,
    config: DetectorConfig = DetectorConfig(),
    domain_guard: DomainGuard | None = None,
) -> SafetyDetection:
    context: list[tuple] = []
    ranked = filter_geometric_detection(
        detect_fast(cloud, config, context_out=context), model
    )
    if domain_guard is None:
        return ranked
    return apply_domain_guard(
        cloud,
        ranked,
        model,
        domain_guard,
        config,
        context[0] if context else None,
    )


def filter_geometric_detection(
    geometric: SafetyDetection,
    model: RiskModel,
) -> SafetyDetection:
    """Apply the learned infrastructure filter to existing geometry output."""
    scored = [(model.score(item), item) for item in geometric.obstacles]
    accepted = [(score, item) for score, item in scored if score >= model.threshold]
    if accepted:
        score, nearest = min(accepted, key=lambda pair: pair[1].distance_min_m)
        return replace(
            geometric,
            state="OBSTACLE",
            obstacle=True,
            nearest_distance_m=nearest.distance_min_m,
            confidence=score,
            obstacles=tuple(item for _, item in accepted),
            reason="geometry_plus_learned_infrastructure_rejection",
        )
    if geometric.obstacles:
        best = max((score for score, _ in scored), default=0.0)
        return replace(
            geometric,
            state="UNKNOWN",
            obstacle=False,
            confidence=best,
            obstacles=(),
            reason="geometric_component_rejected_by_risk_model",
        )
    return geometric
