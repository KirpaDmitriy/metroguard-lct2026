import unittest

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from lidar_geometry.portable_trees import PortableExtraTrees


class PortableExtraTreesTest(unittest.TestCase):
    def test_matches_sklearn_probabilities(self):
        rng = np.random.default_rng(17)
        train = rng.normal(size=(80, 4))
        labels = (train[:, 0] + train[:, 1] * train[:, 2] > 0).astype(int)
        model = ExtraTreesClassifier(
            n_estimators=11, max_depth=4, random_state=19
        ).fit(train, labels)
        test = rng.normal(size=(25, 4))

        portable = PortableExtraTrees.from_sklearn(model)

        np.testing.assert_allclose(
            portable.score(test), model.predict_proba(test)[:, 1], atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
