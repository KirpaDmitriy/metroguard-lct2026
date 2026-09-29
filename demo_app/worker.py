from __future__ import annotations

import argparse
import sqlite3
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from demo_app.algorithms import Detector
from demo_app.config import ROOT
from demo_app.job_store import write_json
from demo_app.visualization import compact_cloud
from lidar_geometry.pointcloud2 import iter_bag_messages


def inspect_bag(path: Path) -> int:
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"topics", "messages"}.issubset(tables):
            raise ValueError("SQLite file is not a rosbag2 database")
        pointcloud_topics = connection.execute(
            "SELECT COUNT(*) FROM topics WHERE type = ?",
            ("sensor_msgs/msg/PointCloud2",),
        ).fetchone()[0]
        if not pointcloud_topics:
            raise ValueError("Bag has no sensor_msgs/msg/PointCloud2 topic")
        return int(
            connection.execute(
                """
            SELECT COUNT(*) FROM messages
            JOIN topics ON topics.id = messages.topic_id
            WHERE topics.type = ?
            """,
                ("sensor_msgs/msg/PointCloud2",),
            ).fetchone()[0]
        )


def analyze_bag(
    input_path: Path,
    algorithm: str,
    progress_path: Path,
    max_frames: int | None = None,
    include_visualization: bool = True,
) -> dict:
    total = inspect_bag(input_path)
    if max_frames is not None:
        total = min(total, max_frames)
    detector = Detector(algorithm, ROOT)
    states: Counter[str] = Counter()
    timeline = []
    latencies = []
    started = time.monotonic()
    for frame, (timestamp, cloud) in enumerate(iter_bag_messages(input_path)):
        if max_frames is not None and frame >= max_frames:
            break
        frame_started = time.perf_counter()
        detection, context = detector.analyze(cloud)
        latency_ms = (time.perf_counter() - frame_started) * 1000
        latencies.append(latency_ms)
        states[detection.state] += 1
        item = {
            "frame": frame,
            "timestamp_ns": timestamp,
            "state": detection.state,
            "distance_m": detection.nearest_distance_m,
            "confidence": detection.confidence,
            "observability": detection.observability,
            "latency_ms": latency_ms,
            "reason": detection.reason,
            "obstacles": [asdict(item) for item in detection.obstacles[:3]],
        }
        if include_visualization:
            visualization = compact_cloud(context, detection.obstacles)
            if visualization is not None:
                item["visualization"] = visualization
        timeline.append(item)
        if frame % 5 == 0 or frame + 1 == total:
            write_json(
                progress_path,
                {
                    "status": "running",
                    "processed_frames": frame + 1,
                    "total_frames": total,
                    "progress": (frame + 1) / max(1, total),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    elapsed = time.monotonic() - started
    return {
        "algorithm": algorithm,
        "summary": {
            "frames": len(timeline),
            "states": dict(states),
            "obstacle_frames": states["OBSTACLE"],
            "nearest_obstacle_m": min(
                (
                    item["distance_m"]
                    for item in timeline
                    if item["state"] == "OBSTACLE"
                ),
                default=None,
            ),
            "elapsed_seconds": elapsed,
            "throughput_fps": len(timeline) / elapsed if elapsed else 0.0,
            "latency_p50_ms": statistics.median(latencies) if latencies else None,
            "latency_p95_ms": (
                sorted(latencies)[round(0.95 * (len(latencies) - 1))]
                if latencies
                else None
            ),
        },
        "timeline": timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--without-visualization", action="store_true")
    args = parser.parse_args()
    result = analyze_bag(
        args.input,
        args.algorithm,
        args.progress,
        args.max_frames,
        include_visualization=not args.without_visualization,
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
