"""Run the frozen MVP on the organisers' official synthetic rosbag."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.temporal import ComponentTracker


def episodes(rows: list[dict], key: str) -> list[dict]:
    result: list[dict] = []
    active: list[dict] = []
    for row in rows + [{"frame": -1, key: "END"}]:
        if row[key] == "OBSTACLE":
            active.append(row)
        elif active:
            distances = [r[f"{key}_distance_m"] for r in active]
            result.append({
                "first_frame": active[0]["frame"],
                "last_frame": active[-1]["frame"],
                "frames": len(active),
                "start_s": active[0]["time_s"],
                "end_s": active[-1]["time_s"],
                "nearest_m": min(x for x in distances if x is not None),
                "median_m": statistics.median(x for x in distances if x is not None),
            })
            active = []
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--model", type=Path, default=Path("lidar_geometry/models/risk_model.json"))
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--guard-band", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("lidar_geometry/artifacts/official_synthetic_frozen.json"))
    args = parser.parse_args()
    model = RiskModel.load(args.model)
    config = DetectorConfig(component_guard_band_m=args.guard_band)
    tracker = ComponentTracker()
    rows: list[dict] = []
    latencies: list[float] = []
    first_timestamp: int | None = None
    schema = None
    for frame, (timestamp, cloud) in enumerate(iter_bag_messages(args.bag)):
        if frame % args.every:
            continue
        if first_timestamp is None:
            first_timestamp = timestamp
            schema = {
                "point_step": cloud.point_step,
                "fields": [field.name for field in cloud.fields],
                "points_per_first_frame": cloud.point_count,
            }
        started = time.perf_counter()
        geometric = detect_fast(cloud, config)
        hybrid = filter_geometric_detection(geometric, model)
        tracked = tracker.update(hybrid)
        latencies.append((time.perf_counter() - started) * 1000)
        rows.append({
            "frame": frame,
            "time_s": round((timestamp - first_timestamp) / 1e9, 4),
            "geometry": geometric.state,
            "geometry_distance_m": geometric.nearest_distance_m,
            "geometry_confidence": geometric.confidence,
            "geometry_reason": geometric.reason,
            "hybrid": hybrid.state,
            "hybrid_distance_m": hybrid.nearest_distance_m,
            "hybrid_confidence": hybrid.confidence,
            "hybrid_reason": hybrid.reason,
            "tracked": tracked.state,
            "tracked_distance_m": tracked.nearest_distance_m,
            "tracked_confidence": tracked.confidence,
            "tracked_reason": tracked.reason,
            "components": [asdict(item) for item in geometric.obstacles],
        })
    report = {
        "protocol": "official-bag audit; bag was previously inspected and is not a blind test",
        "model": str(args.model),
        "bag": str(args.bag),
        "schema": schema,
        "sample_every": args.every,
        "component_guard_band_m": args.guard_band,
        "frames": len(rows),
        "duration_s": rows[-1]["time_s"] if rows else 0,
        "geometry_states": Counter(row["geometry"] for row in rows),
        "hybrid_states": Counter(row["hybrid"] for row in rows),
        "tracked_states": Counter(row["tracked"] for row in rows),
        "geometry_episodes": episodes(rows, "geometry"),
        "hybrid_episodes": episodes(rows, "hybrid"),
        "tracked_episodes": episodes(rows, "tracked"),
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
        },
        "frames_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frames_detail"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
