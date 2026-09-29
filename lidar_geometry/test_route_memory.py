import unittest

import numpy as np

from lidar_geometry.route_memory_experiment import (
    aligned_change_score,
    change_score,
    coherent_aligned_change_score,
    clearance_coherent_change_score,
)


class RouteMemoryTest(unittest.TestCase):
    def test_positive_change_scores_above_identical_patch(self):
        reference = np.zeros((3, 16, 16), dtype=np.float32)
        current = reference.copy()
        current[:, 7:10, 7:10] = 0.8
        self.assertEqual(change_score(reference, reference), 0)
        self.assertGreater(change_score(reference, current), 0)

    def test_removed_structure_does_not_create_obstacle_score(self):
        reference = np.ones((3, 16, 16), dtype=np.float32)
        current = reference.copy()
        current[:, 7:10, 7:10] = 0
        self.assertEqual(change_score(reference, current), 0)

    def test_alignment_removes_one_cell_registration_error(self):
        reference = np.zeros((3, 16, 16), dtype=np.float32)
        reference[:, 2:5, 2:5] = 0.7
        current = np.roll(reference, 1, axis=2)
        self.assertEqual(aligned_change_score(reference, current), 0)

    def test_coherent_score_requires_residual_and_density(self):
        reference = np.zeros((3, 16, 16), dtype=np.float32)
        current = reference.copy()
        current[0, 7:10, 7:10] = 0.8
        self.assertEqual(coherent_aligned_change_score(reference, current), 0)
        current[2, 7:10, 7:10] = 0.8
        self.assertGreater(coherent_aligned_change_score(reference, current), 0)

    def test_clearance_gate_rejects_ceiling_only_change(self):
        reference = np.zeros((3, 16, 16), dtype=np.float32)
        current = reference.copy()
        current[0, 7:10, 7:10] = 0.8
        current[1, 7:10, 7:10] = 1.0
        current[2, 7:10, 7:10] = 0.8
        self.assertEqual(clearance_coherent_change_score(reference, current), 0)
        current[1, 8, 8] = 0.5
        self.assertGreater(clearance_coherent_change_score(reference, current), 0)


if __name__ == "__main__":
    unittest.main()
