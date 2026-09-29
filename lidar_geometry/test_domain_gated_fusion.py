import tempfile
import unittest
from pathlib import Path

import numpy as np

from lidar_geometry.domain_gated_fusion import domain_confidence
from lidar_geometry.domain_guard import DomainGuard, context_descriptor


class DomainGatedFusionTest(unittest.TestCase):
    def test_context_signature_ignores_candidate_center(self):
        patches = np.zeros((2, 3, 16, 16), dtype=np.float32)
        patches[1, :, 4:12, 4:12] = 1
        np.testing.assert_array_equal(
            context_descriptor(patches[:1]),
            context_descriptor(patches[1:]),
        )

    def test_unseen_context_gets_lower_confidence(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(0, 0.1, (20, 24))
        context = np.vstack((reference, np.zeros((1, 24)), np.full((1, 24), 5.0)))
        train_real = np.zeros(22, dtype=bool)
        train_real[:20] = True
        train = train_real.copy()
        test = ~train
        _, confidence, _ = domain_confidence(context, train_real, train, test)
        self.assertGreater(confidence[0], confidence[1])

    def test_fitted_guard_round_trip(self):
        rng = np.random.default_rng(11)
        patches = rng.uniform(0, 0.2, (12, 3, 16, 16)).astype(np.float32)
        guard = DomainGuard.fit(patches)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guard.npz"
            guard.save(path)
            restored = DomainGuard.load(path)
            np.testing.assert_allclose(
                guard.confidence(patches[:2]),
                restored.confidence(patches[:2]),
                rtol=1e-6,
            )
