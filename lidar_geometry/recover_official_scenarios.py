from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.fast_detector import (
    _fast_track_profile,
    _interpolate_track,
    cloud_arrays,
)
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.scenario_catalog import SCENARIOS


BIN_M = 0.25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--anchor", type=float, default=100.0)
    parser.add_argument("--spacing", type=float, default=100.0)
    parser.add_argument("--radius", type=float, default=75.0)
    parser.add_argument("--output", type=Path, default=Path(
        "lidar_geometry/artifacts/official_scenario_positions.json"
    ))
    args = parser.parse_args()
    motion = json.loads(args.motion.read_text(encoding="utf-8"))["frames_detail"]
    config = DetectorConfig()
    edges = []
    counts = []
    frame_support = []
    for index in range(10):
        nominal = args.anchor + index * args.spacing
        scenario_edges = np.arange(
            nominal - args.radius, nominal + args.radius + BIN_M, BIN_M
        )
        edges.append(scenario_edges)
        counts.append(np.zeros(len(scenario_edges) - 1, dtype=np.int64))
        frame_support.append([set() for _ in range(len(scenario_edges) - 1)])

    for frame, (_, cloud) in enumerate(iter_bag_messages(args.bag)):
        arrays = cloud_arrays(cloud)
        distance = -arrays["y"]
        raw_lateral = arrays["x"]
        z = arrays["z"]
        intensity = arrays["intensity"]
        valid = (
            np.isfinite(distance) & np.isfinite(raw_lateral) & np.isfinite(z)
            & (distance >= 3.0) & (distance <= 120.0)
            & (np.abs(raw_lateral) <= config.path_search_half_width_m)
        )
        profile, _ = _fast_track_profile(
            distance[valid], raw_lateral[valid], z[valid], config
        )
        center, rail_z, supported = _interpolate_track(distance, profile, config)
        lateral = raw_lateral - center
        height = z - rail_z
        world = motion[frame]["cumulative_m"] + distance
        synthetic = supported & (intensity == 1)
        for index, scenario in enumerate(SCENARIOS[:10]):
            target = scenario.obstacle
            half_width = target.width_m / 2
            mask = (
                synthetic
                & (lateral >= target.lateral_m - half_width)
                & (lateral <= target.lateral_m + half_width)
                & (height >= target.bottom_m)
                & (height <= target.bottom_m + target.height_m)
            )
            selected = world[mask]
            histogram = np.histogram(selected, edges[index])[0]
            counts[index] += histogram
            for bin_index in np.flatnonzero(histogram):
                frame_support[index][int(bin_index)].add(frame)

    recovered = []
    for index, scenario in enumerate(SCENARIOS[:10]):
        width_bins = max(1, round(scenario.obstacle.length_m / BIN_M))
        kernel = np.ones(width_bins, dtype=np.int64)
        smoothed = np.convolve(counts[index], kernel, mode="same")
        ranked = np.argsort(smoothed)[::-1][:10]
        candidates = []
        for bin_index in ranked:
            center = (edges[index][bin_index] + edges[index][bin_index + 1]) / 2
            lo = max(0, bin_index - width_bins // 2)
            hi = min(len(frame_support[index]), lo + width_bins)
            frames = set().union(*frame_support[index][lo:hi])
            candidates.append({
                "world_m": round(float(center), 3),
                "synthetic_intensity_points": int(smoothed[bin_index]),
                "support_frames": len(frames),
                "first_frame": min(frames) if frames else None,
                "last_frame": max(frames) if frames else None,
            })
        recovered.append({
            "scenario": scenario.name,
            "expected_alarm": scenario.expected_alarm,
            "nominal_world_m": args.anchor + index * args.spacing,
            "candidates": candidates,
        })
    report = {
        "protocol": {
            "runtime_feature": False,
            "purpose": "forensic pseudo-label recovery for the organizer synthetic bag",
            "evidence": "rail-relative stated bounds plus intensity exactly 1 and self-odometry",
            "bin_m": BIN_M,
            "search_radius_m": args.radius,
        },
        "scenarios": recovered,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        row["scenario"]: row["candidates"][:3] for row in recovered
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
