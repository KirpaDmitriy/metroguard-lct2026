from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demo_app.algorithms import Detector as DemoDetector
from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.fast_detector import SafetyDetection
from lidar_geometry.runtime_detector import RuntimeDetector


ROOT = Path(__file__).resolve().parents[1]


def geometric_detection() -> SafetyDetection:
    obstacle = Obstacle(
        points=40,
        voxels=8,
        distance_min_m=20.0,
        distance_max_m=20.5,
        lateral_min_m=-0.3,
        lateral_max_m=0.3,
        height_min_m=0.1,
        height_max_m=1.8,
        surface_residual_max_m=1.7,
        surface_residual_mean_m=0.8,
        intensity_mean=25.0,
    )
    return SafetyDetection(
        state="OBSTACLE",
        obstacle=True,
        nearest_distance_m=20.0,
        confidence=0.75,
        observability=1.0,
        candidate_points=40,
        track_bins=20,
        observed_track_bins=20,
        reliable_range_min_m=3.0,
        reliable_range_max_m=100.0,
        obstacles=(obstacle,),
        reason="geometric_obstacle",
    )


class RuntimeDetectorTest(unittest.TestCase):
    def test_demo_and_shared_tree_runtime_are_exactly_equivalent(self):
        shared = RuntimeDetector.from_root("tree_hybrid", ROOT)
        demo = DemoDetector("tree_hybrid", ROOT)
        geometric = geometric_detection()
        with patch(
            "lidar_geometry.runtime_detector.detect_fast",
            return_value=geometric,
        ):
            self.assertEqual(demo(None), shared(None))

    def test_missing_default_tree_model_is_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                RuntimeDetector.from_root("tree_hybrid", Path(directory))

    def test_geometry_does_not_require_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = RuntimeDetector.from_root("geometry", Path(directory))
            geometric = geometric_detection()
            with patch(
                "lidar_geometry.runtime_detector.detect_fast",
                return_value=geometric,
            ):
                self.assertIs(detector(None), geometric)


if __name__ == "__main__":
    unittest.main()
