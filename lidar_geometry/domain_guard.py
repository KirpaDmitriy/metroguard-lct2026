from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lidar_geometry.detect_obstacles import DetectorConfig, Obstacle
from lidar_geometry.fast_detector import (
    SafetyDetection,
    _fast_track_profile,
    _interpolate_track,
    _surface_reference,
    cloud_arrays,
)
from lidar_geometry.pointcloud2 import PointCloud2
from lidar_geometry.risk_model import RiskModel


GRID = 16
FORWARD_SPAN_M = 3.2
LATERAL_SPAN_M = 2.4


def track_coordinates(cloud: PointCloud2, config: DetectorConfig) -> tuple[np.ndarray, ...] | None:
    arrays = cloud_arrays(cloud)
    distance = -arrays["y"]
    valid = (
        np.isfinite(arrays["x"])
        & np.isfinite(arrays["y"])
        & np.isfinite(arrays["z"])
        & ((arrays["x"] != 0) | (arrays["y"] != 0) | (arrays["z"] != 0))
        & (distance >= config.min_range_m)
        & (distance <= config.max_range_m)
        & (np.abs(arrays["x"]) <= config.path_search_half_width_m)
    )
    distance = distance[valid]
    lateral_raw = arrays["x"][valid]
    z = arrays["z"][valid]
    profile, _ = _fast_track_profile(distance, lateral_raw, z, config)
    if not profile:
        return None
    center, rail_z, supported = _interpolate_track(distance, profile, config)
    distance = distance[supported]
    lateral = lateral_raw[supported] - center[supported]
    z = z[supported]
    rail_z = rail_z[supported]
    inside = np.abs(lateral) <= config.half_width_m + 0.15
    distance, lateral, z, rail_z = (
        value[inside] for value in (distance, lateral, z, rail_z)
    )
    surface = _surface_reference(
        distance,
        lateral,
        z,
        rail_z,
        config,
        config.half_width_m + 0.15,
    )
    return distance, lateral, z - rail_z, z - surface


def component_patch(coordinates: tuple[np.ndarray, ...], item: Obstacle) -> np.ndarray:
    distance, lateral, height, residual = coordinates
    center = (item.distance_min_m + item.distance_max_m) / 2
    forward_min = center - FORWARD_SPAN_M / 2
    lateral_min = -LATERAL_SPAN_M / 2
    row = np.floor((distance - forward_min) * GRID / FORWARD_SPAN_M).astype(np.int32)
    column = np.floor((lateral - lateral_min) * GRID / LATERAL_SPAN_M).astype(np.int32)
    keep = (
        (row >= 0) & (row < GRID)
        & (column >= 0) & (column < GRID)
        & (height >= -0.3) & (height <= 3.1)
    )
    row, column = row[keep], column[keep]
    height, residual = height[keep], residual[keep]
    flat = row * GRID + column
    count = np.bincount(flat, minlength=GRID * GRID).astype(np.float32)
    max_height = np.zeros(GRID * GRID, dtype=np.float32)
    max_residual = np.zeros(GRID * GRID, dtype=np.float32)
    if len(flat):
        np.maximum.at(max_height, flat, np.maximum(height, 0))
        np.maximum.at(max_residual, flat, np.maximum(residual, 0))
    density = np.log1p(count)
    density /= max(1.0, float(density.max()))
    return np.stack((
        density.reshape(GRID, GRID),
        np.clip(max_height.reshape(GRID, GRID) / 3.0, 0, 1),
        np.clip(max_residual.reshape(GRID, GRID), 0, 1),
    ))


def context_descriptor(patches: np.ndarray) -> np.ndarray:
    patches = patches.copy()
    patches[:, :, :, (0, 15)] = 0
    border = np.ones((GRID, GRID), dtype=bool)
    border[4:12, 4:12] = False
    context = patches[:, :, border]
    features = [
        context.mean(axis=2),
        context.std(axis=2),
        context.max(axis=2),
        np.quantile(context, 0.75, axis=2),
    ]
    for rows, columns in (
        (slice(0, 8), slice(0, 8)),
        (slice(0, 8), slice(8, 16)),
        (slice(8, 16), slice(0, 8)),
        (slice(8, 16), slice(8, 16)),
    ):
        local_border = border[rows, columns]
        features.append(patches[:, :, rows, columns][:, :, local_border].mean(axis=2))
    return np.column_stack(features).astype(np.float64)


def logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1 - 1e-6)
    return float(np.log(clipped / (1 - clipped)))


@dataclass(frozen=True)
class DomainGuard:
    median: np.ndarray
    scale: np.ndarray
    references: np.ndarray
    radius: float
    confidence_threshold: float = 0.25
    strong_margin: float = 1.0

    @classmethod
    def fit(cls, patches: np.ndarray) -> "DomainGuard":
        descriptor = context_descriptor(patches)
        median = np.median(descriptor, axis=0)
        scale = np.median(np.abs(descriptor - median), axis=0) * 1.4826
        scale[scale < 1e-4] = 1.0
        references = np.clip((descriptor - median) / scale, -8, 8)
        nearest = np.empty(len(references))
        for start in range(0, len(references), 256):
            block = references[start:start + 256]
            distance = ((block[:, None, :] - references[None, :, :]) ** 2).mean(axis=2)
            rows = np.arange(len(block))
            distance[rows, start + rows] = np.inf
            nearest[start:start + len(block)] = distance.min(axis=1)
        return cls(median, scale, references, max(float(np.quantile(nearest, 0.95)), 1e-6))

    def confidence(self, patches: np.ndarray) -> np.ndarray:
        descriptor = context_descriptor(patches)
        values = np.clip((descriptor - self.median) / self.scale, -8, 8)
        nearest = np.empty(len(values))
        for start in range(0, len(values), 256):
            block = values[start:start + 256]
            distance = ((block[:, None, :] - self.references[None, :, :]) ** 2).mean(axis=2)
            nearest[start:start + len(block)] = distance.min(axis=1)
        return np.exp(-np.log(2) * nearest / self.radius)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            median=self.median,
            scale=self.scale,
            references=self.references.astype(np.float32),
            radius=np.asarray(self.radius),
            confidence_threshold=np.asarray(self.confidence_threshold),
            strong_margin=np.asarray(self.strong_margin),
        )

    @classmethod
    def load(cls, path: Path) -> "DomainGuard":
        data = np.load(path, allow_pickle=False)
        return cls(
            data["median"], data["scale"], data["references"],
            float(data["radius"]), float(data["confidence_threshold"]),
            float(data["strong_margin"]),
        )


def apply_domain_guard(
    cloud: PointCloud2,
    detection: SafetyDetection,
    risk_model: RiskModel,
    guard: DomainGuard,
    config: DetectorConfig,
    coordinates: tuple[np.ndarray, ...] | None = None,
) -> SafetyDetection:
    if detection.state != "OBSTACLE" or not detection.obstacles:
        return detection
    if coordinates is None:
        coordinates = track_coordinates(cloud, config)
    if coordinates is None:
        return detection
    patches = np.asarray([component_patch(coordinates, item) for item in detection.obstacles])
    familiarity = guard.confidence(patches)
    accepted = []
    for item, domain_score in zip(detection.obstacles, familiarity):
        risk_score = risk_model.score(item)
        strong = logit(risk_score) - logit(risk_model.threshold) >= guard.strong_margin
        if domain_score >= guard.confidence_threshold or strong:
            accepted.append(item)
    if accepted:
        return replace(detection, obstacles=tuple(accepted))
    return replace(
        detection,
        state="UNKNOWN",
        obstacle=False,
        obstacles=(),
        reason="unfamiliar_tunnel_context_requires_independent_evidence",
    )
