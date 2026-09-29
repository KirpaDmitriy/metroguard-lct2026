import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from lidar_geometry.evidence_memory import RangeEvidenceMemory
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.risk_model import component_features
from lidar_geometry.runtime_detector import RuntimeDetector

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[round(fraction * (len(values) - 1))]


def benchmark(bag: Path, algorithm: str) -> dict:
    detector = RuntimeDetector.from_root(algorithm, ROOT)
    latencies = []
    states = []
    for _, cloud in iter_bag_messages(bag):
        started = time.perf_counter()
        detection = detector(cloud)
        latencies.append((time.perf_counter() - started) * 1000)
        states.append(detection.state)
    return {
        "frames": len(states),
        "obstacle_frames": states.count("OBSTACLE"),
        "unknown_frames": states.count("UNKNOWN"),
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies),
    }


def benchmark_memory_update(bag: Path) -> dict:
    detector = RuntimeDetector.from_root("tree_hybrid", ROOT)
    memory = RangeEvidenceMemory(detector.tree.threshold)
    latencies = []
    maximum_score_error = 0.0
    for _, cloud in iter_bag_messages(bag):
        geometric = detect_fast(cloud)
        if geometric.obstacles:
            features = np.vstack(
                [component_features(item) for item in geometric.obstacles]
            )
            scores = detector.tree.score(features)
            vectorized_tree_scores = detector.tree.tree.score(features)
            reference = sequential_score(detector.tree.tree, features)
            maximum_score_error = max(
                maximum_score_error,
                float(np.max(np.abs(vectorized_tree_scores - reference))),
            )
        else:
            scores = np.empty(0)
        started = time.perf_counter()
        memory.update(geometric, scores)
        latencies.append((time.perf_counter() - started) * 1000)
    return {
        "frames": len(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies),
        "maximum_score_error": maximum_score_error,
    }


def sequential_score(tree, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float64)
    rows = np.arange(len(values))
    for start in tree.offsets[:-1]:
        nodes = np.full(len(values), start, dtype=np.int32)
        while True:
            features = tree.feature[nodes]
            active = features >= 0
            if not active.any():
                break
            active_rows = rows[active]
            active_nodes = nodes[active]
            go_left = (
                values[active_rows, features[active]] <= tree.threshold[active_nodes]
            )
            nodes[active] = np.where(
                go_left,
                tree.left[active_nodes],
                tree.right[active_nodes],
            )
        result += tree.positive_probability[nodes]
    return result / (len(tree.offsets) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/memory_runtime_exp075.json"),
    )
    args = parser.parse_args()
    report = {
        "experiment": "EXP-075",
        "protocol": {
            "bag": args.bag.name,
            "comparison": "portable per-frame hybrid versus portable range-evidence memory",
            "training": False,
            "network": False,
        },
        "tree_hybrid": benchmark(args.bag, "tree_hybrid"),
        "memory_hybrid": benchmark(args.bag, "memory_hybrid"),
        "incremental_memory_update": benchmark_memory_update(args.bag),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
