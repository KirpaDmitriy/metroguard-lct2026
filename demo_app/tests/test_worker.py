from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from demo_app.worker import inspect_bag


def create_bag(path: Path, topic_type: str = "sensor_msgs/msg/PointCloud2") -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT);
            CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
        """
        )
        connection.execute(
            "INSERT INTO topics(id, name, type) VALUES(1, '/lidar', ?)",
            (topic_type,),
        )
        connection.execute(
            "INSERT INTO messages(topic_id, timestamp, data) VALUES(1, 1, X'00')"
        )


class WorkerTest(unittest.TestCase):
    def test_inspects_rosbag_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.db3"
            create_bag(path)
            self.assertEqual(inspect_bag(path), 1)

    def test_rejects_database_without_pointcloud(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.db3"
            create_bag(path, "std_msgs/msg/String")
            with self.assertRaisesRegex(ValueError, "PointCloud2"):
                inspect_bag(path)
