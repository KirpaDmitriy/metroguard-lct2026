import unittest

import numpy as np

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.risk_model import FEATURE_NAMES, RiskModel, component_features, fit_logistic


def obstacle(points=20, height=0.5):
    return Obstacle(
        points=points, voxels=4, distance_min_m=20.0, distance_max_m=20.4,
        lateral_min_m=-0.2, lateral_max_m=0.2, height_min_m=0.05,
        height_max_m=height, surface_residual_max_m=height,
        surface_residual_mean_m=height / 2, intensity_mean=40.0,
    )


class RiskModelTest(unittest.TestCase):
    def test_feature_schema_is_finite(self):
        vector = component_features(obstacle())
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(vector).all())

    def test_fit_round_trip(self):
        low = component_features(obstacle(points=5, height=0.08))
        high = component_features(obstacle(points=80, height=1.0))
        x = np.vstack([low, low * 1.01, high, high * 1.01])
        y = np.asarray([0, 0, 1, 1])
        model = fit_logistic(x, y, steps=400)
        self.assertGreater(model.score(obstacle(points=80, height=1.0)),
                           model.score(obstacle(points=5, height=0.08)))


if __name__ == "__main__":
    unittest.main()
