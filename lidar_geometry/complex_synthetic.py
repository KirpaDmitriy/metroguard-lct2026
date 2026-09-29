from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.pointcloud2 import PointCloud2
from lidar_geometry.synthetic import InjectionResult, SyntheticObject, inject_box


@dataclass(frozen=True)
class ComplexScenario:
    name: str
    expected_alarm: bool
    envelope: SyntheticObject
    parts: tuple[SyntheticObject, ...]


def scenario_library(distance_m: float) -> tuple[ComplexScenario, ...]:
    def part(
        distance_offset: float, lateral: float, width: float, length: float,
        height: float, bottom: float,
    ) -> SyntheticObject:
        return SyntheticObject(
            "box", distance_m + distance_offset, lateral, width, length, height,
            bottom_m=bottom,
        )

    return (
        ComplexScenario(
            "person_silhouette", True,
            SyntheticObject("composite", distance_m, 0.0, 0.48, 0.34, 1.72),
            (
                part(0, 0, 0.42, 0.30, 0.92, 0.55),
                part(0, 0, 0.24, 0.24, 0.25, 1.47),
                part(0, -0.13, 0.14, 0.26, 0.55, 0),
                part(0, 0.13, 0.14, 0.26, 0.55, 0),
            ),
        ),
        ComplexScenario(
            "l_shaped_debris", True,
            SyntheticObject("composite", distance_m, 0.35, 0.90, 0.55, 0.85),
            (
                part(0, 0.35, 0.90, 0.55, 0.18, 0),
                part(0, 0.70, 0.20, 0.55, 0.85, 0),
            ),
        ),
        ComplexScenario(
            "forked_cable", True,
            SyntheticObject("composite", distance_m, 0.0, 0.30, 0.12, 2.0, bottom_m=1.0),
            (
                part(0, 0, 0.05, 0.08, 1.55, 1.0),
                part(0, -0.10, 0.20, 0.08, 0.06, 1.72),
                part(0, 0.10, 0.20, 0.08, 0.06, 2.15),
            ),
        ),
        ComplexScenario(
            "outside_irregular_stack", False,
            SyntheticObject("composite", distance_m, 1.70, 0.70, 0.65, 1.20),
            (
                part(-0.12, 1.55, 0.40, 0.40, 0.55, 0),
                part(0.12, 1.82, 0.38, 0.42, 0.85, 0),
                part(0, 1.68, 0.25, 0.25, 0.35, 0.85),
            ),
        ),
        ComplexScenario(
            "overhead_bracket", False,
            SyntheticObject("composite", distance_m, 0.0, 1.60, 0.35, 0.45, bottom_m=3.05),
            (
                part(0, 0, 1.60, 0.30, 0.12, 3.38),
                part(0, -0.70, 0.20, 0.30, 0.45, 3.05),
                part(0, 0.70, 0.20, 0.30, 0.45, 3.05),
            ),
        ),
    )


def inject_composite(
    cloud: PointCloud2,
    scenario: ComplexScenario,
    config: DetectorConfig = DetectorConfig(),
    seed: int = 0,
) -> InjectionResult:
    current = cloud
    indices: set[int] = set()
    bounds_min = np.full(3, np.inf)
    bounds_max = np.full(3, -np.inf)
    center = rail_z = 0.0
    for index, obstacle in enumerate(scenario.parts):
        result = inject_box(current, obstacle, config, seed + index)
        current = result.cloud
        indices.update(result.modified_indices)
        bounds_min = np.minimum(bounds_min, result.object_bounds_xyz[0])
        bounds_max = np.maximum(bounds_max, result.object_bounds_xyz[1])
        center = result.track_center_m
        rail_z = result.rail_z_m
    return InjectionResult(
        current,
        tuple(sorted(indices)),
        (tuple(bounds_min.tolist()), tuple(bounds_max.tolist())),
        center,
        rail_z,
    )
