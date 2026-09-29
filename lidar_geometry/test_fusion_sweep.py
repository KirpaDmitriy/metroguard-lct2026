from __future__ import annotations

import unittest

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.fusion_sweep import obvious_intrusion


def component(lateral_min: float, lateral_max: float, height_max: float) -> Obstacle:
    return Obstacle(
        points=30,
        voxels=10,
        distance_min_m=20,
        distance_max_m=20.2,
        lateral_min_m=lateral_min,
        lateral_max_m=lateral_max,
        height_min_m=0,
        height_max_m=height_max,
        surface_residual_max_m=height_max,
        surface_residual_mean_m=height_max / 2,
        intensity_mean=20,
    )


class FusionSweepTest(unittest.TestCase):
    def test_large_central_intrusion_bypasses_ranker(self):
        self.assertTrue(obvious_intrusion(component(-0.9, 0.9, 0.2), 1.0))
        self.assertTrue(obvious_intrusion(component(-0.05, 0.05, 1.2), 1.0))

    def test_large_outside_component_does_not_bypass_ranker(self):
        self.assertFalse(obvious_intrusion(component(0.8, 2.0, 1.5), 1.0))


if __name__ == "__main__":
    unittest.main()
