from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lidar_geometry.detect_obstacles import Obstacle


def _inside_obstacle(
    distance: np.ndarray,
    lateral: np.ndarray,
    height: np.ndarray,
    obstacle: Obstacle | dict,
) -> np.ndarray:
    value = obstacle if isinstance(obstacle, dict) else obstacle.__dict__
    return (
        (distance >= value["distance_min_m"] - 0.08)
        & (distance <= value["distance_max_m"] + 0.08)
        & (lateral >= value["lateral_min_m"] - 0.08)
        & (lateral <= value["lateral_max_m"] + 0.08)
        & (height >= value["height_min_m"] - 0.08)
        & (height <= value["height_max_m"] + 0.08)
    )


def _even_sample(indexes: np.ndarray, limit: int) -> np.ndarray:
    if len(indexes) <= limit:
        return indexes
    positions = np.linspace(0, len(indexes) - 1, limit, dtype=np.int64)
    return indexes[positions]


def compact_cloud(
    context: tuple[np.ndarray, ...] | None,
    obstacles: Iterable[Obstacle | dict],
    *,
    background_limit: int = 420,
    obstacle_limit: int = 180,
) -> dict | None:
    if context is None or len(context) < 3:
        return None
    distance, lateral, height = context[:3]
    valid = (
        np.isfinite(distance)
        & np.isfinite(lateral)
        & np.isfinite(height)
        & (distance >= 0)
        & (distance <= 150)
        & (np.abs(lateral) <= 2.4)
        & (height >= -0.25)
        & (height <= 3.25)
    )
    if not np.any(valid):
        return None
    selected = np.zeros(len(distance), dtype=bool)
    for obstacle in obstacles:
        selected |= _inside_obstacle(distance, lateral, height, obstacle)
    selected &= valid
    background = valid & ~selected
    chosen = np.concatenate((
        _even_sample(np.flatnonzero(background), background_limit),
        _even_sample(np.flatnonzero(selected), obstacle_limit),
    ))
    points = [
        [round(float(lateral[index]), 2), round(float(distance[index]), 1),
         round(float(height[index]), 2), int(selected[index])]
        for index in chosen
    ]
    return {"points": points}
