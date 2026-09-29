from __future__ import annotations

import unittest

import numpy as np

from demo_app.visualization import compact_cloud


class VisualizationTest(unittest.TestCase):
    def test_keeps_obstacle_points_when_background_is_limited(self):
        distance = np.arange(12, dtype=np.float64)
        lateral = np.zeros(12, dtype=np.float64)
        height = np.zeros(12, dtype=np.float64)
        obstacle = {
            "distance_min_m": 8.0,
            "distance_max_m": 9.0,
            "lateral_min_m": -0.1,
            "lateral_max_m": 0.1,
            "height_min_m": -0.1,
            "height_max_m": 0.1,
        }
        result = compact_cloud(
            (distance, lateral, height),
            [obstacle],
            background_limit=2,
            obstacle_limit=10,
        )
        self.assertIsNotNone(result)
        highlighted = [point for point in result["points"] if point[3]]
        self.assertEqual([point[1] for point in highlighted], [8.0, 9.0])

    def test_rejects_non_finite_and_out_of_view_points(self):
        context = (
            np.array([10.0, np.nan, 151.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        result = compact_cloud(context, [])
        self.assertEqual(result["points"], [[0.0, 10.0, 1.0, 0]])
