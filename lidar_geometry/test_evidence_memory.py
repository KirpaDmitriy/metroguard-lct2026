import unittest

import numpy as np

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.evidence_memory import RangeEvidenceMemory, WorldEvidenceMemory
from lidar_geometry.fast_detector import SafetyDetection


def detection(*items: Obstacle) -> SafetyDetection:
    return SafetyDetection(
        state="OBSTACLE" if items else "CLEAR",
        obstacle=bool(items),
        nearest_distance_m=items[0].distance_min_m if items else None,
        confidence=0.9 if items else 0.0,
        observability=1.0,
        candidate_points=10 * len(items),
        track_bins=20,
        observed_track_bins=20,
        reliable_range_min_m=3.0,
        reliable_range_max_m=100.0,
        obstacles=items,
        reason="test",
    )


def obstacle(distance: float, lateral: float = 0.0) -> Obstacle:
    return Obstacle(
        points=10,
        voxels=2,
        distance_min_m=distance,
        distance_max_m=distance + 0.2,
        lateral_min_m=lateral - 0.1,
        lateral_max_m=lateral + 0.1,
        height_min_m=0.1,
        height_max_m=0.8,
        surface_residual_max_m=0.8,
        surface_residual_mean_m=0.4,
        intensity_mean=1.0,
    )


class WorldEvidenceMemoryTest(unittest.TestCase):
    def test_confirms_matching_subthreshold_support(self):
        memory = WorldEvidenceMemory(0.8)
        first = memory.update(detection(obstacle(40.0)), np.asarray([0.82]), 0.0)
        second = memory.update(detection(obstacle(39.0)), np.asarray([0.65]), 1.0)
        self.assertEqual(first.state, "UNKNOWN")
        self.assertEqual(second.state, "OBSTACLE")
        self.assertEqual(second.reason, "online_evidence_memory")

    def test_rejects_isolated_high_score(self):
        memory = WorldEvidenceMemory(0.8)
        result = memory.update(detection(obstacle(40.0)), np.asarray([0.95]), 0.0)
        self.assertEqual(result.state, "UNKNOWN")

    def test_does_not_match_different_lateral_position(self):
        memory = WorldEvidenceMemory(0.8)
        memory.update(detection(obstacle(40.0, -0.8)), np.asarray([0.9]), 0.0)
        result = memory.update(detection(obstacle(39.0, 0.8)), np.asarray([0.9]), 1.0)
        self.assertEqual(result.state, "UNKNOWN")

    def test_rejects_score_length_mismatch(self):
        memory = WorldEvidenceMemory(0.8)
        with self.assertRaisesRegex(ValueError, "counts differ"):
            memory.update(detection(obstacle(40.0)), np.asarray([]), 0.0)

    def test_empty_detection_stays_clear(self):
        memory = WorldEvidenceMemory(0.8)
        result = memory.update(detection(), np.asarray([]), 0.0)
        self.assertEqual(result.state, "CLEAR")

    def test_range_memory_tracks_approaching_component(self):
        memory = RangeEvidenceMemory(0.8)
        first = memory.update(detection(obstacle(40.0)), np.asarray([0.82]))
        second = memory.update(detection(obstacle(37.8)), np.asarray([0.65]))
        self.assertEqual(first.state, "UNKNOWN")
        self.assertEqual(second.state, "OBSTACLE")
