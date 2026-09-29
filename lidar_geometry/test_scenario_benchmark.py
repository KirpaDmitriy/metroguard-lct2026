from __future__ import annotations

import unittest

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.scenario_benchmark import target_alarm
from lidar_geometry.scenario_catalog import matches_target
from lidar_geometry.synthetic import SyntheticObject


def component(distance: float, lateral: float) -> Obstacle:
    return Obstacle(
        points=10,
        voxels=4,
        distance_min_m=distance - 0.1,
        distance_max_m=distance + 0.1,
        lateral_min_m=lateral - 0.1,
        lateral_max_m=lateral + 0.1,
        height_min_m=0.0,
        height_max_m=0.2,
        surface_residual_max_m=0.2,
        surface_residual_mean_m=0.15,
        intensity_mean=20.0,
    )


class ScenarioBenchmarkTest(unittest.TestCase):
    def test_outside_component_is_not_attributed_across_clearance_boundary(self):
        item = Obstacle(
            points=8,
            voxels=2,
            distance_min_m=19.9,
            distance_max_m=20.1,
            lateral_min_m=0.90,
            lateral_max_m=1.05,
            height_min_m=0.1,
            height_max_m=0.2,
            surface_residual_max_m=0.2,
            surface_residual_mean_m=0.1,
            intensity_mean=1.0,
        )
        outside = SyntheticObject("box", 20, 1.21, 0.3, 0.3, 0.1)
        self.assertFalse(matches_target(item, outside))

    def test_target_matching_rejects_unrelated_component(self):
        obstacle = SyntheticObject("box", 20, 0, 0.3, 0.3, 0.1)
        self.assertTrue(matches_target(component(20, 0), obstacle))
        self.assertFalse(matches_target(component(30, 0), obstacle))
        self.assertFalse(matches_target(component(20, 2), obstacle))

    def test_guard_band_candidate_is_not_attributed_as_alarm(self):
        obstacle = SyntheticObject("box", 20, 0.9, 0.3, 0.3, 0.1)
        candidate = component(20, 0.9)
        self.assertTrue(matches_target(candidate, obstacle))
        self.assertFalse(target_alarm((candidate,), obstacle))
