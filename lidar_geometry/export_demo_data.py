"""Export compact, browser-friendly evidence from real PointCloud2 frames."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.fast_detector import cloud_arrays, detect_fast
from lidar_geometry.hybrid_detector import detect_hybrid
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.synthetic import SYNTHETIC_LIBRARY, inject_box


def get_frame(path: Path, frame_number: int):
    for frame, (_, cloud) in enumerate(iter_bag_messages(path)):
        if frame == frame_number:
            return cloud
    raise IndexError(f"frame {frame_number} not found")


def compact_points(cloud, seed: int = 7, maximum: int = 18000):
    arrays = cloud_arrays(cloud)
    lateral = arrays["x"].astype(np.float64, copy=False)
    forward = -arrays["y"].astype(np.float64, copy=False)
    height = arrays["z"].astype(np.float64, copy=False)
    valid = (
        np.isfinite(lateral) & np.isfinite(forward) & np.isfinite(height)
        & (forward >= 0) & (forward <= 110)
        & (np.abs(lateral) <= 8) & (height >= -3) & (height <= 6)
    )
    indexes = np.flatnonzero(valid)
    if len(indexes) > maximum:
        indexes = np.random.default_rng(seed).choice(indexes, maximum, replace=False)
    points = np.column_stack((lateral[indexes], forward[indexes], height[indexes]))
    return np.round(points, 3).tolist()


def scene(name, cloud, model):
    result = detect_hybrid(cloud, model) if model else detect_fast(cloud)
    return {
        "name": name,
        "points": compact_points(cloud),
        "decision": {
            "state": result.state,
            "distance_m": result.nearest_distance_m,
            "confidence": result.confidence,
            "observability": result.observability,
            "reason": result.reason,
        },
        "components": [asdict(item) for item in result.obstacles[:12]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--frame", type=int, default=10)
    parser.add_argument("--synthetic-frame", type=int, default=170)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("lidar_geometry/demo/demo-data.json"))
    args = parser.parse_args()
    model = RiskModel.load(args.model) if args.model else None
    cloud = get_frame(args.bag, args.frame)
    synthetic_base = get_frame(args.bag, args.synthetic_frame)
    minimum = next(item for item in SYNTHETIC_LIBRARY if item.kind == "minimum_box")
    synthetic = inject_box(synthetic_base, replace(minimum, distance_m=20.0),
                           DetectorConfig(), seed=2026)
    payload = {
        "source": str(args.bag),
        "frame": args.frame,
        "coordinate_system": {"lateral": "x", "forward": "-y", "height": "z"},
        "scenes": [
            scene("Реальный проезд: doubleT_obstacle", cloud, model),
            scene("Ray-synthetic: 0,3×0,3×0,1 м @ 20 м", synthetic.cloud, model),
        ],
        "synthetic_visible_points": synthetic.visible_points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":"),
                                      ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
