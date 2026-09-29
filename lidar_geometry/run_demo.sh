#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/rosbag_directory [topic]" >&2
  exit 2
fi

BAG_PATH=$1
TOPIC=${2:-/lidar_points}
IMAGE=${METRO_GUARD_IMAGE:-metro-guard:mvp}

docker build -f lidar_geometry/Dockerfile.mvp -t "$IMAGE" .

docker run --rm --network host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  "$IMAGE" \
  python3 -m lidar_geometry.ros2_mvp_node --ros-args -p input_topic:="$TOPIC" &
NODE_PID=$!
trap 'kill "$NODE_PID" 2>/dev/null || true' EXIT

source /opt/ros/humble/setup.bash
ros2 bag play "$BAG_PATH"
