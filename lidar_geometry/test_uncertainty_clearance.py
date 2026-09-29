from __future__ import annotations

import unittest

from lidar_geometry.detect_obstacles import Obstacle, TrackSample
from lidar_geometry.fast_detector import SafetyDetection
from lidar_geometry.uncertainty_clearance import (
    apply_uncertainty_gate,
    center_uncertainty_m,
    confidently_in_clearance,
)


def component(lateral_min: float, lateral_max: float) -> Obstacle:
    return Obstacle(
        points=12,
        voxels=4,
        distance_min_m=19.8,
        distance_max_m=20.2,
        lateral_min_m=lateral_min,
        lateral_max_m=lateral_max,
        height_min_m=0.1,
        height_max_m=0.4,
        surface_residual_max_m=0.4,
        surface_residual_mean_m=0.2,
        intensity_mean=10.0,
    )


def detection(item: Obstacle) -> SafetyDetection:
    return SafetyDetection(
        state="OBSTACLE",
        obstacle=True,
        nearest_distance_m=item.distance_min_m,
        confidence=0.9,
        observability=1.0,
        candidate_points=item.points,
        track_bins=20,
        observed_track_bins=20,
        reliable_range_min_m=2.0,
        reliable_range_max_m=80.0,
        obstacles=(item,),
        reason="test",
    )


class UncertaintyClearanceTest(unittest.TestCase):
    def test_linear_rail_fit_has_quantization_floor(self):
        profile = {
            index: TrackSample(0.01 * index, 0.0, 1.52, 10.0)
            for index in range(8, 14)
        }
        self.assertAlmostEqual(center_uncertainty_m(profile, 20.0), 0.025)

    def test_sparse_fit_is_uncertain(self):
        profile = {10: TrackSample(0.0, 0.0, 1.52, 10.0)}
        self.assertEqual(center_uncertainty_m(profile, 20.0), 0.20)

    def test_confidently_inside_component_is_never_downgraded(self):
        profile = {10: TrackSample(0.0, 0.0, 1.52, 10.0)}
        result = apply_uncertainty_gate(detection(component(-0.2, 0.2)), profile)
        self.assertEqual(result.state, "OBSTACLE")

    def test_boundary_component_becomes_unknown_but_is_preserved(self):
        profile = {10: TrackSample(0.0, 0.0, 1.52, 10.0)}
        item = component(0.90, 1.04)
        result = apply_uncertainty_gate(detection(item), profile)
        self.assertEqual(result.state, "UNKNOWN")
        self.assertFalse(result.obstacle)
        self.assertEqual(result.obstacles, (item,))

    def test_inside_and_boundary_clearance_bounds(self):
        self.assertTrue(confidently_in_clearance(component(0.70, 0.90), 0.10, 1.05))
        self.assertFalse(confidently_in_clearance(component(0.98, 1.04), 0.10, 1.05))

    def test_bad_uncertainty_is_rejected(self):
        with self.assertRaises(ValueError):
            confidently_in_clearance(component(0.0, 0.1), float("nan"), 1.05)


if __name__ == "__main__":
    unittest.main()
