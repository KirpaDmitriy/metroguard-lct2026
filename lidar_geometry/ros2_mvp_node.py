"""ROS 2 Humble wrapper for the vectorized MetroGuard detector."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.ego_motion import EgoMotionEstimator, MotionEstimate
from lidar_geometry.pointcloud2 import PointCloud2, PointField
from lidar_geometry.runtime_detector import ALGORITHMS, RuntimeDetector
from lidar_geometry.temporal import (
    ComponentTracker,
    TemporalDecision,
    WorldComponentTracker,
)


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2 as RosPointCloud2
    from std_msgs.msg import String

    class MetroGuardNode(Node):
        def __init__(self):
            super().__init__("metro_guard")
            self.declare_parameter("input_topic", "/lidar_points")
            self.declare_parameter("half_width_m", 1.05)
            self.declare_parameter("clearance_height_m", 3.0)
            self.declare_parameter("min_range_m", 3.0)
            self.declare_parameter("max_range_m", 150.0)
            self.declare_parameter("min_height_m", 0.06)
            self.declare_parameter("algorithm", "memory_hybrid")
            self.declare_parameter(
                "risk_model_path",
                "/opt/metro-guard/lidar_geometry/models/risk_model.json",
            )
            self.declare_parameter(
                "portable_model_path",
                "/opt/metro-guard/lidar_geometry/artifacts/linear_extra_trees_hybrid.npz",
            )
            self.declare_parameter("risk_threshold", -1.0)
            self.declare_parameter(
                "domain_guard_path",
                "/opt/metro-guard/lidar_geometry/models/domain_guard.npz",
            )
            self.declare_parameter("use_ego_motion", True)
            self.declare_parameter("input_reliability", "reliable")
            self.config = DetectorConfig(
                half_width_m=float(self.get_parameter("half_width_m").value),
                clearance_height_m=float(
                    self.get_parameter("clearance_height_m").value
                ),
                min_range_m=float(self.get_parameter("min_range_m").value),
                max_range_m=float(self.get_parameter("max_range_m").value),
                min_height_above_rail_m=float(self.get_parameter("min_height_m").value),
            )
            self.range_tracker = ComponentTracker()
            self.world_tracker = WorldComponentTracker()
            self.motion = EgoMotionEstimator()
            self.use_ego_motion = bool(self.get_parameter("use_ego_motion").value)
            algorithm = str(self.get_parameter("algorithm").value)
            if algorithm not in ALGORITHMS:
                raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
            model_path = Path(str(self.get_parameter("risk_model_path").value))
            portable_path = Path(str(self.get_parameter("portable_model_path").value))
            domain_path = Path(str(self.get_parameter("domain_guard_path").value))
            threshold = float(self.get_parameter("risk_threshold").value)
            self.detector = RuntimeDetector(
                algorithm,
                config=self.config,
                risk_model_path=model_path,
                portable_model_path=portable_path,
                domain_guard_path=domain_path,
                risk_threshold=threshold,
            )
            self.algorithm = algorithm
            self.publisher = self.create_publisher(String, "/metro_guard/detection", 10)
            topic = str(self.get_parameter("input_topic").value)
            reliability = str(self.get_parameter("input_reliability").value)
            if reliability not in {"reliable", "best_effort"}:
                raise ValueError("input_reliability must be reliable or best_effort")
            input_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=(
                    ReliabilityPolicy.RELIABLE
                    if reliability == "reliable"
                    else ReliabilityPolicy.BEST_EFFORT
                ),
                durability=DurabilityPolicy.VOLATILE,
            )
            self.subscription = self.create_subscription(
                RosPointCloud2, topic, self.on_cloud, input_qos
            )
            self.get_logger().info(
                f"MetroGuard listening on {topic} ({algorithm}, {reliability})"
            )

        def on_cloud(self, message: RosPointCloud2) -> None:
            started = time.perf_counter()
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
            raw = self.detector(cloud)
            if self.algorithm == "memory_hybrid":
                motion = MotionEstimate(0.0, 0.0, 0.0, False)
                decision = TemporalDecision(
                    raw.state,
                    raw.obstacle,
                    raw.nearest_distance_m,
                    raw.confidence,
                    2 if raw.obstacle else 0,
                    raw.reason,
                )
            else:
                range_decision = self.range_tracker.update(raw)
                motion = self.motion.update(cloud)
                world_decision = self.world_tracker.update(raw, motion.cumulative_m)
                decision = (
                    world_decision
                    if self.use_ego_motion and motion.reliable
                    else range_decision
                )
            payload = {
                "stamp": {
                    "sec": message.header.stamp.sec,
                    "nanosec": message.header.stamp.nanosec,
                },
                "state": decision.state,
                "obstacle": decision.obstacle,
                "nearest_distance_m": decision.nearest_distance_m,
                "confidence": decision.confidence,
                "confirmations": decision.confirmations,
                "reason": decision.reason,
                "observability": raw.observability,
                "reliable_range_m": [
                    raw.reliable_range_min_m,
                    raw.reliable_range_max_m,
                ],
                "raw_state": raw.state,
                "ego_motion": {
                    "delta_m": motion.delta_m,
                    "cumulative_m": motion.cumulative_m,
                    "correlation": motion.correlation,
                    "reliable": motion.reliable,
                },
                "components": [asdict(item) for item in raw.obstacles[:5]],
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
            self.publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    rclpy.init()
    node = MetroGuardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
