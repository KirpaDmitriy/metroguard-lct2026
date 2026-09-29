from __future__ import annotations

import unittest
from dataclasses import replace

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.evidence_fusion import (
    EvidenceFusion,
    cable_rescue_geometry,
    critical_geometry,
    physical_rescue_geometry,
)
from lidar_geometry.fast_detector import SafetyDetection
from lidar_geometry.risk_model import RiskModel


def component(forward: float, lateral: float, height: float) -> Obstacle:
    return Obstacle(
        points=20,
        voxels=5,
        distance_min_m=20,
        distance_max_m=20 + forward,
        lateral_min_m=-lateral / 2,
        lateral_max_m=lateral / 2,
        height_min_m=0.1,
        height_max_m=0.1 + height,
        surface_residual_max_m=height,
        surface_residual_mean_m=height / 2,
        intensity_mean=20,
    )


class EvidenceFusionTest(unittest.TestCase):
    def test_compact_and_thin_vertical_candidates_use_slow_lane(self):
        self.assertTrue(critical_geometry(component(0.3, 0.3, 0.1)))
        self.assertTrue(critical_geometry(component(0.1, 0.05, 1.5)))

    def test_large_infrastructure_shape_cannot_use_slow_lane(self):
        self.assertFalse(critical_geometry(component(1.5, 1.2, 2.0)))

    def test_physical_rescue_rules_are_shape_specific(self):
        rail_object = component(0.3, 0.3, 0.1)
        rail_object = replace(
            rail_object,
            lateral_min_m=0.60,
            lateral_max_m=0.90,
            surface_residual_mean_m=0.18,
        )
        cable = component(0.05, 0.05, 1.0)
        self.assertTrue(physical_rescue_geometry(rail_object, 0.4))
        self.assertTrue(cable_rescue_geometry(cable, 0.6))
        self.assertFalse(physical_rescue_geometry(component(0.4, 0.4, 0.2), 0.8))

    def test_world_fusion_requires_motion(self):
        model = RiskModel(("log_points",), (0.0,), (1.0,), (0.0,), 0.0, 0.9)
        fusion = EvidenceFusion(model, world_coordinates=True)
        detection = SafetyDetection(
            "UNKNOWN", False, None, 0.0, 0.0, 0, 0, 0, None, None, (), "test"
        )
        with self.assertRaisesRegex(ValueError, "cumulative_m"):
            fusion.update(detection)
