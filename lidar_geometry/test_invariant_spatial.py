import unittest

import numpy as np

from lidar_geometry.invariant_spatial_experiment import (
    invariant_descriptor,
    random_convolution_descriptor,
    randomized_patches,
    transplant_real_object,
)


class InvariantSpatialTest(unittest.TestCase):
    def test_descriptor_shape_and_finiteness(self):
        patches = np.zeros((2, 3, 16, 16), dtype=np.float32)
        patches[1, :, 6:10, 6:10] = 1
        descriptor = invariant_descriptor(patches)
        self.assertEqual(descriptor.shape, (2, 198))
        self.assertTrue(np.isfinite(descriptor).all())
        self.assertGreater(np.linalg.norm(descriptor[1] - descriptor[0]), 0)

    def test_random_convolution_is_deterministic(self):
        rng = np.random.default_rng(9)
        patches = rng.uniform(0, 1, (2, 3, 16, 16)).astype(np.float32)
        first = random_convolution_descriptor(patches)
        second = random_convolution_descriptor(patches)
        self.assertEqual(first.shape, (2, 288))
        np.testing.assert_array_equal(first, second)

    def test_randomization_is_deterministic_and_bounded(self):
        patches = np.full((2, 3, 16, 16), 0.5, dtype=np.float32)
        first = randomized_patches(patches, 4)
        second = randomized_patches(patches, 4)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(np.linalg.norm(first - patches), 0)
        self.assertGreaterEqual(float(first.min()), 0)
        self.assertLessEqual(float(first.max()), 1)

    def test_transplant_changes_only_supported_object_region(self):
        donor = np.zeros((1, 3, 16, 16), dtype=np.float32)
        donor[0, :, 7:9, 7:9] = 0.8
        background = np.zeros((2, 3, 16, 16), dtype=np.float32)
        result = transplant_real_object(donor, background, seed=3, copies=1)
        self.assertEqual(result.shape, (1, 3, 16, 16))
        self.assertGreater(float(result.max()), 0)
        self.assertEqual(float(result[:, :, :3, :3].max()), 0)
