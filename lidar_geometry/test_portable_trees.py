import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from lidar_geometry.portable_trees import PortableExtraTrees, PortableFarRangeRanker


class PortableExtraTreesTest(unittest.TestCase):
    def test_matches_sklearn_probabilities(self):
        rng = np.random.default_rng(17)
        train = rng.normal(size=(80, 4))
        labels = (train[:, 0] + train[:, 1] * train[:, 2] > 0).astype(int)
        model = ExtraTreesClassifier(n_estimators=11, max_depth=4, random_state=19).fit(
            train, labels
        )
        test = rng.normal(size=(25, 4))

        portable = PortableExtraTrees.from_sklearn(model)

        np.testing.assert_allclose(
            portable.score(test), model.predict_proba(test)[:, 1], atol=1e-12
        )

    def test_far_range_ranker_round_trip(self):
        rng = np.random.default_rng(31)
        train = rng.normal(size=(60, 3))
        labels = (train[:, 0] > 0).astype(int)
        model = ExtraTreesClassifier(n_estimators=7, random_state=7).fit(train, labels)
        ranker = PortableFarRangeRanker(
            PortableExtraTrees.from_sklearn(model),
            0.73,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            ranker.save(path)
            restored = PortableFarRangeRanker.load(path)
        self.assertEqual(restored.threshold, 0.73)
        np.testing.assert_allclose(restored.score(train), ranker.score(train))
