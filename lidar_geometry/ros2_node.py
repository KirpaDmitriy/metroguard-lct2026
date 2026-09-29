"""ROS 2 wrapper around the geometric detector.

Publishes compact JSON to ``/obstacle_detection`` for easy inspection with
``ros2 topic echo``.  Imports are local so offline bag processing does not need
ROS installed.
"""

from __future__ import annotations

from dataclasses import asdict
import json

from lidar_geometry.detect_obstacles import DetectorConfig, detect
from lidar_geometry.pointcloud2 import PointCloud2, PointField


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2 as RosPointCloud2
    from std_msgs.msg import String

    class GeometricObstacleNode(Node):
        def __init__(self):
            super().__init__("geometric_obstacle_detector")
            self.declare_parameter("input_topic", "/lidar_points")
            self.declare_parameter("half_width_m", 1.05)
            self.declare_parameter("min_range_m", 3.0)
            self.declare_parameter("max_range_m", 150.0)
            self.declare_parameter("min_height_m", 0.06)
            self.declare_parameter("clearance_height_m", 3.0)
            self.publisher = self.create_publisher(String, "/obstacle_detection", 10)
            topic = self.get_parameter("input_topic").value
            self.subscription = self.create_subscription(RosPointCloud2, topic, self.on_cloud, 10)

        def on_cloud(self, message: RosPointCloud2) -> None:
            cloud = PointCloud2(
                stamp_sec=message.header.stamp.sec,
                stamp_nanosec=message.header.stamp.nanosec,
                frame_id=message.header.frame_id,
                height=message.height,
                width=message.width,
                fields=tuple(
                    PointField(field.name, field.offset, field.datatype, field.count)
                    for field in message.fields
                ),
                is_bigendian=message.is_bigendian,
                point_step=message.point_step,
                row_step=message.row_step,
                data=memoryview(message.data),
                is_dense=message.is_dense,
            )
            config = DetectorConfig(
                half_width_m=float(self.get_parameter("half_width_m").value),
                min_range_m=float(self.get_parameter("min_range_m").value),
                max_range_m=float(self.get_parameter("max_range_m").value),
                min_height_above_rail_m=float(self.get_parameter("min_height_m").value),
                clearance_height_m=float(self.get_parameter("clearance_height_m").value),
            )
            result = detect(cloud, config)
            payload = {
                "obstacle": result.obstacle,
                "nearest_distance_m": result.nearest_distance_m,
                "reliable_range_m": [
                    result.reliable_range_min_m,
                    result.reliable_range_max_m,
                ],
                "clusters": [asdict(item) for item in result.obstacles[:5]],
            }
            self.publisher.publish(String(data=json.dumps(payload)))

    rclpy.init()
    node = GeometricObstacleNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
