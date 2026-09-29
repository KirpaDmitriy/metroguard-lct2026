import unittest

import numpy as np

from lidar_geometry.ego_motion import longitudinal_shift


class EgoMotionTest(unittest.TestCase):
    def test_recovers_forward_shift(self):
        rng = np.random.default_rng(7)
        previous = rng.normal(size=500)
        current = np.r_[previous[13:], np.zeros(13)]
        distance, score = longitudinal_shift(previous, current, max_shift_bins=20)
        self.assertAlmostEqual(distance, 1.3)
        self.assertGreater(score, 0.99)


if __name__ == "__main__":
    unittest.main()
