from __future__ import annotations

import unittest

import numpy as np

from lidar_geometry.evaluate_osdar_transfer import metro_cloud, target_bounds
from lidar_geometry.fast_detector import cloud_arrays


class OsdarTransferTest(unittest.TestCase):
    def test_coordinate_adapter(self):
        values = {
            "x": np.array([10.0], dtype=np.float32),
            "y": np.array([2.0], dtype=np.float32),
            "z": np.array([-1.0], dtype=np.float32),
            "intensity": np.array([7.0], dtype=np.float32),
            "sensor_index": np.array([0.0], dtype=np.float32),
        }
        arrays = cloud_arrays(metro_cloud(values, pandar_only=True))
        self.assertEqual(float(arrays["x"][0]), 2.0)
        self.assertEqual(float(arrays["y"][0]), -10.0)
        self.assertEqual(float(arrays["z"][0]), -1.0)

    def test_target_bounds_are_forward_and_lateral(self):
        values = {
            "x": np.array([20.0, 21.0, 22.0], dtype=np.float32),
            "y": np.array([-0.5, 0.0, 0.5], dtype=np.float32),
            "label": np.ones(3, dtype=np.float32),
        }
        minimum, maximum = target_bounds(values)
        np.testing.assert_array_equal(minimum, [20.0, -0.5])
        np.testing.assert_array_equal(maximum, [22.0, 0.5])


if __name__ == "__main__":
    unittest.main()
