#!/usr/bin/env python3
"""Capture a fixed number of MetroGuard decisions for integration tests."""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    rclpy.init()
    node = Node("metro_guard_capture")
    captured: list[dict] = []

    def on_message(message: String) -> None:
        captured.append(json.loads(message.data))

    subscription = node.create_subscription(
        String,
        "/metro_guard/detection",
        on_message,
        qos_profile_sensor_data,
    )
    deadline = time.monotonic() + args.timeout
    while len(captured) < args.count and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.25)

    node.destroy_subscription(subscription)
    node.destroy_node()
    rclpy.shutdown()

    if len(captured) != args.count:
        raise SystemExit(f"received {len(captured)}/{args.count} decisions")
    print(json.dumps(captured, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
