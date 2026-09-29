from __future__ import annotations

import struct
import unittest
import warnings

import numpy as np

from lidar_geometry.detect_obstacles import CandidatePoint, DetectorConfig, _cluster
from lidar_geometry.fast_detector import (
    _minimum_component_points,
    _nanmedian_last_axis,
    _neighbourhood_3x3,
    cloud_arrays,
    detect_fast,
)
from lidar_geometry.pointcloud2 import PointCloud2, PointField


FIELDS = (
    PointField("x", 0, 7, 1),
    PointField("y", 4, 7, 1),
    PointField("z", 8, 7, 1),
    PointField("intensity", 12, 7, 1),
    PointField("ring", 16, 4, 1),
    PointField("timestamp", 18, 8, 1),
)
PACKER = struct.Struct("<ffffHd")
XYZI_FIELDS = FIELDS[:4]
XYZI_PACKER = struct.Struct("<ffff")


def cloud(points):
    data = b"".join(PACKER.pack(*point) for point in points)
    return PointCloud2(0, 0, "test", 1, len(points), FIELDS, False, 26, len(data), memoryview(data), False)


def xyzi_cloud(points):
    data = b"".join(XYZI_PACKER.pack(*point) for point in points)
    return PointCloud2(
        0, 0, "test", 1, len(points), XYZI_FIELDS, False, 16, len(data),
        memoryview(data), False,
    )


def rail_scene(with_obstacle: bool):
    points = []
    for step in range(40, 601, 2):
        distance = step / 10
        center = 0.003 * distance
        rail_z = -1.5
        for side in (-1, 1):
            for repeat in range(4):
                points.append((
                    center + side * 0.76 + (repeat - 1.5) * 0.015,
                    -distance,
                    rail_z + (repeat % 2) * 0.01,
                    8.0,
                    repeat,
                    0.0,
                ))
        for lateral_step in range(-10, 11):
            points.append((
                center + lateral_step * 0.1,
                -distance,
                rail_z - 0.18,
                3.0,
                lateral_step + 32,
                0.0,
            ))
    if with_obstacle:
        distance = 30.0
        center = 0.003 * distance
        for ix in range(7):
            for iy in range(7):
                points.append((
                    center - 0.15 + ix * 0.05,
                    -(distance - 0.15 + iy * 0.05),
                    -1.4,
                    35.0,
                    ix,
                    0.0,
                ))
    return cloud(points)


class FastDetectorTest(unittest.TestCase):
    def test_inverse_square_support_relaxes_only_distant_components(self):
        self.assertEqual(_minimum_component_points(30.0, "frozen"), 6)
        self.assertEqual(_minimum_component_points(30.0, "inverse_square"), 6)
        self.assertEqual(_minimum_component_points(60.0, "frozen"), 5)
        self.assertEqual(_minimum_component_points(60.0, "inverse_square"), 3)
        self.assertEqual(_minimum_component_points(100.0, "inverse_square"), 3)

    def test_unknown_support_model_fails_visibly(self):
        with self.assertRaisesRegex(ValueError, "unknown support model"):
            _minimum_component_points(60.0, "typo")

    def test_neighbourhood_matches_direct_sum(self):
        histogram = np.arange(35).reshape(5, 7)
        padded = np.pad(histogram, 1)
        expected = sum(
            padded[1 + dx:1 + dx + 5, 1 + dz:1 + dz + 7]
            for dx in (-1, 0, 1)
            for dz in (-1, 0, 1)
        )
        np.testing.assert_array_equal(_neighbourhood_3x3(histogram), expected)

    def test_nanmedian_window_matches_numpy(self):
        values = np.array([
            [np.nan, 1.0, 4.0, 2.0],
            [np.nan, np.nan, np.nan, np.nan],
            [3.0, 1.0, np.nan, 2.0],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            expected = np.nanmedian(values, axis=-1)
        np.testing.assert_allclose(
            _nanmedian_last_axis(values), expected, equal_nan=True
        )

    def test_component_spanning_clearance_is_not_erased_as_wall(self):
        points = []
        for lateral_step in range(-10, 11):
            for height_step in range(1, 11):
                points.append(CandidatePoint(
                    distance=20.0,
                    lateral=lateral_step / 10,
                    z=-1.5 + height_step / 10,
                    height=height_step / 10,
                    surface_residual=height_step / 10,
                    intensity=30.0,
                    ring=height_step,
                ))
        result = _cluster(points, DetectorConfig())
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].lateral_max_m - result[0].lateral_min_m, 1.9)

    def test_zero_copy_fields_have_expected_values(self):
        sample = cloud([(1.0, -2.0, 3.0, 4.0, 5, 6.0)])
        arrays = cloud_arrays(sample)
        self.assertEqual(float(arrays["x"][0]), 1.0)
        self.assertEqual(float(arrays["y"][0]), -2.0)
        self.assertEqual(int(arrays["ring"][0]), 5)

    def test_xyzi_cloud_without_ring_is_supported(self):
        arrays = cloud_arrays(xyzi_cloud([(1.0, -2.0, 3.0, 4.0)]))
        self.assertEqual(float(arrays["x"][0]), 1.0)
        self.assertEqual(int(arrays["ring"][0]), -1)

    def test_three_state_contract_and_minimum_box(self):
        config = DetectorConfig(
            max_range_m=70,
            floor_min_points_per_bin=4,
            path_min_cell_neighbourhood=2,
            min_cluster_points=6,
        )
        clear = detect_fast(rail_scene(False), config)
        self.assertEqual(clear.state, "CLEAR")
        detected = detect_fast(rail_scene(True), config)
        self.assertEqual(detected.state, "OBSTACLE")
        self.assertAlmostEqual(detected.nearest_distance_m or 0, 29.85, delta=0.3)

    def test_continuity_track_model_preserves_simple_scene(self):
        config = DetectorConfig(
            max_range_m=70,
            floor_min_points_per_bin=4,
            path_min_cell_neighbourhood=2,
            min_cluster_points=6,
        )
        greedy = detect_fast(rail_scene(True), config, track_model="greedy")
        continuity = detect_fast(rail_scene(True), config, track_model="continuity")
        self.assertEqual(continuity.state, greedy.state)
        self.assertAlmostEqual(
            continuity.nearest_distance_m or 0,
            greedy.nearest_distance_m or 0,
            delta=0.1,
        )

    def test_unknown_track_model_fails_visibly(self):
        with self.assertRaisesRegex(ValueError, "unknown track model"):
            detect_fast(rail_scene(False), track_model="typo")

    def test_unobservable_frame_is_unknown(self):
        result = detect_fast(cloud([(0.0, -10.0, -1.0, 2.0, 1, 0.0)]))
        self.assertEqual(result.state, "UNKNOWN")
        self.assertFalse(result.obstacle)

    def test_empty_and_nonfinite_frames_are_unknown(self):
        empty = detect_fast(cloud([]))
        nonfinite = detect_fast(cloud([(float("nan"), -10.0, -1.0, 2.0, 1, 0.0)]))
        self.assertEqual(empty.state, "UNKNOWN")
        self.assertEqual(nonfinite.state, "UNKNOWN")

    def test_missing_required_field_fails_visibly(self):
        malformed = PointCloud2(
            0, 0, "test", 1, 0, FIELDS[:3], False, 16, 0, memoryview(b""), False
        )
        with self.assertRaisesRegex(ValueError, "intensity"):
            detect_fast(malformed)


if __name__ == "__main__":
    unittest.main()
