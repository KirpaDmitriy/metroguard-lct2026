from __future__ import annotations

import unittest

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.fast_detector import SafetyDetection
from lidar_geometry.portable_temporal_experiment import (
    count_episodes,
    selected_detection,
)


def obstacle(distance: float) -> Obstacle:
    return Obstacle(
        points=10,
        voxels=4,
        distance_min_m=distance,
        distance_max_m=distance + 0.5,
        lateral_min_m=-0.2,
        lateral_max_m=0.2,
        height_min_m=0.1,
        height_max_m=0.8,
        surface_residual_max_m=0.3,
        surface_residual_mean_m=0.2,
        intensity_mean=20.0,
    )


def geometric_detection(*items: Obstacle) -> SafetyDetection:
    return SafetyDetection(
        state="OBSTACLE",
        obstacle=True,
        nearest_distance_m=min(item.distance_min_m for item in items),
        confidence=0.7,
        observability=0.9,
        candidate_points=20,
        track_bins=20,
        observed_track_bins=18,
        reliable_range_min_m=3.0,
        reliable_range_max_m=100.0,
        obstacles=items,
        reason="geometry",
    )


class PortableTemporalExperimentTest(unittest.TestCase):
    def test_selected_detection_uses_nearest_accepted_component(self):
        near = obstacle(12.0)
        far = obstacle(30.0)
        result = selected_detection(geometric_detection(near, far), (far,), (0.93,))
        self.assertTrue(result.obstacle)
        self.assertEqual(result.nearest_distance_m, 30.0)
        self.assertEqual(result.obstacles, (far,))

    def test_rejected_components_do_not_remain_obstacles(self):
        result = selected_detection(geometric_detection(obstacle(12.0)), (), ())
        self.assertFalse(result.obstacle)
        self.assertEqual(result.state, "UNKNOWN")
        self.assertEqual(result.obstacles, ())

    def test_episode_count_uses_boolean_transitions(self):
        self.assertEqual(count_episodes([False, True, True, False, True]), 2)
