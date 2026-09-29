from __future__ import annotations

import struct
import unittest

from lidar_geometry.detect_obstacles import DetectorConfig, detect
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


def cloud(points):
    data = b"".join(PACKER.pack(*point) for point in points)
    return PointCloud2(
        0,
        0,
        "test",
        1,
        len(points),
        FIELDS,
        False,
        26,
        len(data),
        memoryview(data),
        False,
    )


class DetectorTest(unittest.TestCase):
    def test_minimum_size_low_box_is_detected_from_its_top(self):
        ground = []
        for distance in range(4, 41):
            for lateral_step in range(-10, 11):
                ground.append((lateral_step * 0.1, -float(distance), -1.5, 5.0, 0, 0.0))

        box_top = []
        for forward_step in range(7):
            for lateral_step in range(7):
                box_top.append(
                    (
                        -0.15 + lateral_step * 0.05,
                        -(19.85 + forward_step * 0.05),
                        -1.40,
                        30.0,
                        lateral_step,
                        0.0,
                    )
                )
        config = DetectorConfig(max_range_m=50, floor_min_points_per_bin=4)
        result = detect(cloud(ground + box_top), config)
        self.assertTrue(result.obstacle)
        self.assertAlmostEqual(result.nearest_distance_m or 0, 19.85, delta=0.1)

    def test_empty_track_is_clear_and_object_is_detected(self):
        ground = []
        for distance in range(4, 41):
            for repeat in range(10):
                lateral = -1.2 + repeat * 0.27
                ground.append((lateral, -float(distance), -1.5, 5.0, repeat, 0.0))

        config = DetectorConfig(
            max_range_m=50,
            floor_min_points_per_bin=4,
            min_cluster_points=6,
            min_cluster_vertical_extent_m=0.1,
        )
        self.assertFalse(detect(cloud(ground), config).obstacle)

        obstacle = []
        for ix in range(5):
            for iz in range(8):
                obstacle.append(
                    (
                        -0.4 + ix * 0.2,
                        -(19.9 + (ix % 2) * 0.08),
                        -0.8 + iz * 0.15,
                        40.0,
                        iz,
                        0.0,
                    )
                )
        result = detect(cloud(ground + obstacle), config)
        self.assertTrue(result.obstacle)
        self.assertAlmostEqual(result.nearest_distance_m or 0, 19.9, delta=0.2)

    def test_curved_track_uses_rail_relative_coordinates(self):
        rails = []
        for step in range(40, 401, 5):
            distance = step / 10
            center = distance * 0.01
            rail_z = -1.5 - distance * 0.002
            for side in (-1, 1):
                for repeat in range(5):
                    lateral = center + side * 0.76 + (repeat - 2) * 0.01
                    rails.append((lateral, -distance, rail_z, 8.0, repeat, 0.0))

        config = DetectorConfig(
            max_range_m=50,
            floor_min_points_per_bin=4,
            min_cluster_points=6,
            min_cluster_vertical_extent_m=0.1,
        )
        clear = detect(cloud(rails), config)
        self.assertFalse(clear.obstacle)
        self.assertGreater(clear.track_bins, 10)

        obstacle = []
        center = 30 * 0.01
        rail_z = -1.5 - 30 * 0.002
        for ix in range(5):
            for iz in range(8):
                obstacle.append(
                    (
                        center - 0.4 + ix * 0.2,
                        -(29.9 + (ix % 2) * 0.08),
                        rail_z + 0.5 + iz * 0.15,
                        40.0,
                        iz,
                        0.0,
                    )
                )
        result = detect(cloud(rails + obstacle), config)
        self.assertTrue(result.obstacle)
        self.assertAlmostEqual(result.nearest_distance_m or 0, 29.9, delta=0.2)
