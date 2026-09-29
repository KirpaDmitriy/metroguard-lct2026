"""Inspect whether a manually selected component persists in nearby frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lidar_geometry.detect_obstacles import detect
from lidar_geometry.ml_experiments import proposal_config
from lidar_geometry.pointcloud2 import iter_bag_messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--frames", default="20:30")
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--lateral", type=float, required=True)
    parser.add_argument("--max-height", type=float, default=0.6)
    args = parser.parse_args()
    start, end = (int(value) for value in args.frames.split(":"))
    config = proposal_config()
    for frame, (_timestamp, cloud) in enumerate(iter_bag_messages(args.bag)):
        if frame < start:
            continue
        if frame > end:
            break
        result = detect(cloud, config)
        matches = []
        for item in result.obstacles:
            center = (item.lateral_min_m + item.lateral_max_m) / 2
            if item.height_max_m > args.max_height:
                continue
            cost = abs(item.distance_min_m - args.distance) + 2 * abs(center - args.lateral)
            matches.append((cost, item))
        matches.sort(key=lambda pair: pair[0])
        best = matches[0][1] if matches and matches[0][0] < 2.0 else None
        print(json.dumps({
            "frame": frame,
            "match": None if best is None else {
                "points": best.points,
                "distance_m": best.distance_min_m,
                "lateral_center_m": (best.lateral_min_m + best.lateral_max_m) / 2,
                "width_m": best.lateral_max_m - best.lateral_min_m,
                "length_m": best.distance_max_m - best.distance_min_m,
                "height_m": best.height_max_m,
            },
        }))


if __name__ == "__main__":
    main()
