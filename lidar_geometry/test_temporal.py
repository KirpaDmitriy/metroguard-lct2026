from __future__ import annotations

import unittest

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.fast_detector import SafetyDetection
from lidar_geometry.temporal import ComponentTracker, WorldComponentTracker


def component(distance: float, lateral: float = 0.0) -> Obstacle:
    return Obstacle(
        points=20,
        voxels=6,
        distance_min_m=distance - 0.1,
        distance_max_m=distance + 0.1,
        lateral_min_m=lateral - 0.1,
        lateral_max_m=lateral + 0.1,
        height_min_m=0.1,
        height_max_m=0.4,
        surface_residual_max_m=0.4,
        surface_residual_mean_m=0.2,
        intensity_mean=20,
    )


def detection(*items: Obstacle) -> SafetyDetection:
    nearest = min((item.distance_min_m for item in items), default=None)
    return SafetyDetection(
        state="OBSTACLE" if items else "CLEAR",
        obstacle=bool(items),
        nearest_distance_m=nearest,
        confidence=0.85,
        observability=0.8,
        candidate_points=20 * len(items),
        track_bins=40,
        observed_track_bins=35,
        reliable_range_min_m=3,
        reliable_range_max_m=80,
        obstacles=items,
        reason="test",
    )


class ComponentTrackerTest(unittest.TestCase):
    def test_world_tracker_uses_ego_motion(self):
        tracker = WorldComponentTracker()
        first = tracker.update(detection(component(40.0)), 0.0)
        second = tracker.update(detection(component(38.0)), 2.0)
        self.assertFalse(first.obstacle)
        self.assertTrue(second.obstacle)

    def test_world_tracker_rejects_range_without_matching_motion(self):
        tracker = WorldComponentTracker()
        tracker.update(detection(component(40.0)), 0.0)
        second = tracker.update(detection(component(40.0)), 2.0)
        self.assertFalse(second.obstacle)

    def test_same_approaching_component_is_confirmed(self):
        tracker = ComponentTracker()
        self.assertFalse(tracker.update(detection(component(40))).obstacle)
        decision = tracker.update(detection(component(38)))
        self.assertTrue(decision.obstacle)
        self.assertEqual(decision.reason, "range_lateral_component_track")

    def test_unrelated_components_are_not_combined(self):
        tracker = ComponentTracker()
        tracker.update(detection(component(40, -0.5)))
        decision = tracker.update(detection(component(20, 0.5)))
        self.assertFalse(decision.obstacle)

    def test_track_expires_after_misses(self):
        tracker = ComponentTracker()
        tracker.update(detection(component(40)))
        tracker.update(detection())
        tracker.update(detection())
        tracker.update(detection())
        self.assertFalse(tracker.update(detection(component(38))).obstacle)

    def test_three_of_five_requires_three_recent_component_hits(self):
        tracker = ComponentTracker(
            required_hits=3,
            max_missed_frames=4,
            confirmation_window_frames=5,
        )
        self.assertFalse(tracker.update(detection(component(40))).obstacle)
        tracker.update(detection())
        self.assertFalse(tracker.update(detection(component(38))).obstacle)
        tracker.update(detection())
        self.assertTrue(tracker.update(detection(component(36))).obstacle)
        tracker.update(detection())
        tracker.update(detection())
        tracker.update(detection())
        self.assertFalse(tracker.update(detection(component(32))).obstacle)

    def test_three_of_five_world_hits_use_rolling_window(self):
        tracker = WorldComponentTracker(
            required_hits=3,
            max_missed_frames=4,
            confirmation_window_frames=5,
        )
        self.assertFalse(tracker.update(detection(component(40)), 0).obstacle)
        tracker.update(detection(), 1)
        self.assertFalse(tracker.update(detection(component(38)), 2).obstacle)
        tracker.update(detection(), 3)
        self.assertTrue(tracker.update(detection(component(36)), 4).obstacle)
