from __future__ import annotations

from dataclasses import replace

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.fast_detector import SafetyDetection
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.temporal import ComponentTracker, TemporalDecision, WorldComponentTracker


def critical_geometry(item: Obstacle) -> bool:
    forward = item.distance_max_m - item.distance_min_m
    lateral = item.lateral_max_m - item.lateral_min_m
    height = item.height_max_m - item.height_min_m
    compact = forward <= 0.8 and lateral <= 0.6 and height <= 0.6
    thin_vertical = forward <= 0.5 and lateral <= 0.25 and height >= 0.8
    return compact or thin_vertical


def cable_rescue_geometry(item: Obstacle, score: float) -> bool:
    forward = item.distance_max_m - item.distance_min_m
    lateral = item.lateral_max_m - item.lateral_min_m
    height = item.height_max_m - item.height_min_m
    lateral_center = abs((item.lateral_min_m + item.lateral_max_m) / 2)
    return (
        forward <= 0.08
        and lateral <= 0.08
        and height >= 0.30
        and lateral_center <= 0.10
        and score >= 0.50
    )


def short_rail_rescue_geometry(item: Obstacle, score: float) -> bool:
    forward = item.distance_max_m - item.distance_min_m
    lateral = item.lateral_max_m - item.lateral_min_m
    lateral_center = abs((item.lateral_min_m + item.lateral_max_m) / 2)
    return (
        forward <= 0.60
        and lateral <= 0.60
        and item.height_max_m <= 0.35
        and item.surface_residual_mean_m >= 0.16
        and lateral_center >= 0.55
        and score >= 0.20
    )


def physical_rescue_geometry(item: Obstacle, score: float) -> bool:
    return cable_rescue_geometry(item, score) or short_rail_rescue_geometry(item, score)


def detection_from_components(
    geometric: SafetyDetection,
    components: tuple[Obstacle, ...],
    confidence: float,
) -> SafetyDetection:
    return replace(
        geometric,
        state="OBSTACLE" if components else (
            "CLEAR" if geometric.state == "CLEAR" else "UNKNOWN"
        ),
        obstacle=bool(components),
        nearest_distance_m=min(
            (item.distance_min_m for item in components),
            default=geometric.nearest_distance_m,
        ),
        confidence=confidence,
        obstacles=components,
    )


class EvidenceFusion:
    def __init__(
        self,
        model: RiskModel,
        low_threshold: float = 0.3,
        max_distance_step_m: float = 3.5,
        world_coordinates: bool = False,
        physical_rules: bool = False,
        cable_only: bool = False,
    ):
        self.model = model
        self.low_threshold = low_threshold
        self.world_coordinates = world_coordinates
        self.physical_rules = physical_rules
        self.cable_only = cable_only
        if world_coordinates:
            self.high_tracker = WorldComponentTracker(required_hits=2)
            self.low_tracker = WorldComponentTracker(required_hits=4)
        else:
            self.high_tracker = ComponentTracker(
                required_hits=2,
                max_distance_step_m=max_distance_step_m,
            )
            self.low_tracker = ComponentTracker(
                required_hits=4,
                max_distance_step_m=max_distance_step_m,
                max_confirmed_velocity_m_per_frame=0.5,
            )

    def _track(self, tracker, detection, cumulative_m):
        if self.world_coordinates:
            if cumulative_m is None:
                raise ValueError("cumulative_m is required for world-coordinate fusion")
            return tracker.update(detection, cumulative_m)
        return tracker.update(detection)

    def update(
        self,
        geometric: SafetyDetection,
        cumulative_m: float | None = None,
    ) -> TemporalDecision:
        high = filter_geometric_detection(geometric, self.model)
        high_decision = self._track(self.high_tracker, high, cumulative_m)
        scored = [(self.model.score(item), item) for item in geometric.obstacles]
        low_scored = [
            (score, item) for score, item in scored
            if (
                cable_rescue_geometry(item, score)
                if self.cable_only else (
                    physical_rescue_geometry(item, score)
                    if self.physical_rules else critical_geometry(item)
                )
            )
        ]
        low_components = tuple(
            item for score, item in low_scored
            if (self.physical_rules or self.cable_only or score >= self.low_threshold)
            and score < self.model.threshold
        )
        low = detection_from_components(
            geometric,
            low_components,
            max((score for score, _ in low_scored), default=0.0),
        )
        low_decision = self._track(self.low_tracker, low, cumulative_m)
        if high_decision.obstacle:
            return high_decision
        if low_decision.obstacle:
            return replace(low_decision, reason="persistent_critical_geometry")
        if high.state == "OBSTACLE" or low_components:
            distances = [
                value for value in (
                    high.nearest_distance_m,
                    low.nearest_distance_m,
                ) if value is not None
            ]
            return TemporalDecision(
                "UNKNOWN",
                False,
                min(distances, default=None),
                max(high.confidence, low.confidence),
                0,
                "awaiting_independent_evidence",
            )
        return TemporalDecision(
            geometric.state,
            False,
            geometric.nearest_distance_m,
            geometric.confidence,
            0,
            geometric.reason,
        )
