from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from lidar_geometry.fast_detector import cloud_arrays
from lidar_geometry.pointcloud2 import PointCloud2, iter_bag_messages


BIN_M = 0.1
RANGE_EDGES_M = np.arange(3.0, 120.0 + BIN_M, BIN_M)


def tunnel_profile(cloud: PointCloud2) -> np.ndarray:
    arrays = cloud_arrays(cloud)
    distance = -arrays["y"]
    lateral = arrays["x"]
    z = arrays["z"]
    wall = (
        (distance >= 3.0) & (distance < 120.0)
        & (np.abs(lateral) >= 1.3) & (np.abs(lateral) <= 4.0)
        & (z >= -2.0) & (z <= 4.0)
    )
    counts = np.histogram(distance[wall], RANGE_EDGES_M)[0].astype(np.float64)
    return np.convolve(np.log1p(counts), np.ones(5) / 5, mode="same")


def longitudinal_shift(
    previous: np.ndarray,
    current: np.ndarray,
    max_shift_bins: int = 30,
) -> tuple[float, float]:
    previous = previous[:-5]
    current = current[:-5]
    length = len(previous)
    shifts = np.arange(max_shift_bins + 1)
    overlap = length - shifts
    previous_sum = np.concatenate(([0.0], np.cumsum(previous)))
    current_sum = np.concatenate(([0.0], np.cumsum(current)))
    previous_square_sum = np.concatenate(([0.0], np.cumsum(previous * previous)))
    current_square_sum = np.concatenate(([0.0], np.cumsum(current * current)))
    left_sum = previous_sum[length] - previous_sum[shifts]
    right_sum = current_sum[overlap]
    left_variance = (
        previous_square_sum[length] - previous_square_sum[shifts]
        - left_sum * left_sum / overlap
    )
    right_variance = current_square_sum[overlap] - right_sum * right_sum / overlap
    dot = np.correlate(previous, current, mode="full")[
        length - 1:length + max_shift_bins
    ]
    covariance = dot - left_sum * right_sum / overlap
    denominator = np.sqrt(np.maximum(0.0, left_variance * right_variance))
    scores = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, -1.0),
        where=denominator > 0,
    )
    best_shift = int(np.argmax(scores))
    return best_shift * BIN_M, float(scores[best_shift])


@dataclass(frozen=True)
class MotionEstimate:
    delta_m: float
    cumulative_m: float
    correlation: float
    reliable: bool


class EgoMotionEstimator:
    def __init__(self, minimum_correlation: float = 0.5):
        self.minimum_correlation = minimum_correlation
        self.previous: np.ndarray | None = None
        self.cumulative_m = 0.0

    def update(self, cloud: PointCloud2) -> MotionEstimate:
        current = tunnel_profile(cloud)
        if self.previous is None:
            delta, correlation, reliable = 0.0, 1.0, False
        else:
            candidate, correlation = longitudinal_shift(self.previous, current)
            reliable = correlation >= self.minimum_correlation
            delta = candidate if reliable else 0.0
        self.cumulative_m += delta
        self.previous = current
        return MotionEstimate(delta, self.cumulative_m, correlation, reliable)


def estimate_motion(bag: Path) -> list[dict]:
    rows = []
    previous = None
    cumulative = 0.0
    first_timestamp = None
    for frame, (timestamp, cloud) in enumerate(iter_bag_messages(bag)):
        profile = tunnel_profile(cloud)
        if first_timestamp is None:
            first_timestamp = timestamp
        delta, quality = (0.0, 1.0) if previous is None else longitudinal_shift(previous, profile)
        cumulative += delta
        rows.append({
            "frame": frame,
            "time_s": (timestamp - first_timestamp) / 1e9,
            "delta_m": delta,
            "cumulative_m": cumulative,
            "correlation": quality,
        })
        previous = profile
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/official_ego_motion.json"),
    )
    args = parser.parse_args()
    rows = estimate_motion(args.bag)
    report = {
        "method": "longitudinal cross-correlation of wall-return range profiles",
        "bin_m": BIN_M,
        "frames": len(rows),
        "distance_m": rows[-1]["cumulative_m"],
        "median_delta_m": float(np.median([row["delta_m"] for row in rows[1:]])),
        "median_correlation": float(np.median([row["correlation"] for row in rows[1:]])),
        "frames_detail": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "frames", "distance_m", "median_delta_m", "median_correlation"
    )}, indent=2))


if __name__ == "__main__":
    main()
