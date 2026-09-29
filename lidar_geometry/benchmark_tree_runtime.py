from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np

from lidar_geometry.classical_hybrid_experiment import extra_trees
from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    linear,
    sample_weights,
    threshold_from_real,
)
from lidar_geometry.evaluate_mvp import selected_frames
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.portable_trees import PortableExtraTrees, PortableLinearTreeHybrid
from lidar_geometry.risk_model import component_features


def percentiles(values: list[float]) -> dict:
    return {
        "p50_ms": float(np.quantile(values, 0.50)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
    }


def evaluate(
    cache: Path, bag: Path, model_path: Path, portable_path: Path, output: Path,
    trees: int,
) -> dict:
    data = np.load(cache, allow_pickle=False)
    features = data["features"].astype(np.float64)
    labels = data["labels"]
    records = json.loads(str(data["records_json"]))
    weight = sample_weights(labels, records)
    linear_model = fit_logistic_features(
        features, labels, weight, linear, steps=1500, l2=0.06
    )
    tree_model = extra_trees(trees)
    tree_model.fit(features, labels, sample_weight=weight)
    linear_scores = linear_model.score(features)
    tree_scores = tree_model.predict_proba(features)[:, 1]
    thresholds = {
        "linear": threshold_from_real(linear_scores, records),
        "extra_trees": threshold_from_real(tree_scores, records),
        "tree_weighted_75": threshold_from_real(
            0.25 * linear_scores + 0.75 * tree_scores, records
        ),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tree_model, model_path, compress=3)
    parameters = linear_model.parameters
    if parameters is None:
        raise RuntimeError("linear predictor is not serializable")
    hybrid_threshold = threshold_from_real(
        0.25 * linear_scores + 0.75 * tree_scores, records
    )
    portable_hybrid = PortableLinearTreeHybrid(
        PortableExtraTrees.from_sklearn(tree_model),
        parameters["raw_mean"], parameters["raw_scale"], parameters["mean"],
        parameters["scale"], parameters["weights"], parameters["bias"],
        0.75, hybrid_threshold,
    )
    portable_hybrid.save(portable_path)
    load_started = time.perf_counter()
    loaded_tree = joblib.load(model_path)
    load_ms = (time.perf_counter() - load_started) * 1000
    portable_load_started = time.perf_counter()
    portable_hybrid = PortableLinearTreeHybrid.load(portable_path)
    portable_tree = portable_hybrid.tree
    portable_load_ms = (time.perf_counter() - portable_load_started) * 1000

    batch = np.tile(features, (50, 1))[:100_000]
    started = time.perf_counter()
    loaded_tree.predict_proba(batch)
    batch_elapsed = time.perf_counter() - started
    portable_started = time.perf_counter()
    portable_scores = portable_tree.score(batch)
    portable_elapsed = time.perf_counter() - portable_started
    np.testing.assert_allclose(
        portable_scores, loaded_tree.predict_proba(batch)[:, 1], atol=1e-12
    )
    expected_hybrid = 0.25 * linear_model.score(batch) + 0.75 * portable_scores
    hybrid_error = float(np.max(np.abs(portable_hybrid.score(batch) - expected_hybrid)))
    np.testing.assert_allclose(
        portable_hybrid.score(batch), expected_hybrid, atol=1e-12
    )

    clouds = [cloud for _, cloud in selected_frames(bag, every=5, maximum=100)]
    modes = (
        "geometry", "linear", "extra_trees", "tree_weighted_75",
        "portable_tree_weighted_75",
    )
    latencies = {name: [] for name in modes}
    candidate_counts = []
    for cloud in clouds[:3]:
        detect_fast(cloud)
    for repeat in range(2):
        for cloud_index, cloud in enumerate(clouds):
            offset = (cloud_index + repeat) % len(modes)
            ordered_modes = modes[offset:] + modes[:offset]
            for name in ordered_modes:
                started = time.perf_counter()
                detection = detect_fast(cloud)
                if detection.obstacles and name != "geometry":
                    matrix = np.vstack([
                        component_features(item) for item in detection.obstacles
                    ])
                    if name == "linear":
                        linear_model.score(matrix)
                    elif name == "extra_trees":
                        loaded_tree.predict_proba(matrix)[:, 1]
                    elif name == "tree_weighted_75":
                        (
                            0.25 * linear_model.score(matrix)
                            + 0.75 * loaded_tree.predict_proba(matrix)[:, 1]
                        )
                    else:
                        portable_hybrid.score(matrix)
                latencies[name].append((time.perf_counter() - started) * 1000)
                if name == "geometry" and repeat == 0:
                    candidate_counts.append(len(detection.obstacles))
    report = {
        "protocol": {
            "frames": len(clouds),
            "bag": str(bag),
            "frame_stride": 5,
            "timed_repeats": 2,
            "mode_order": "rotated per frame and repeat after warm-up",
            "timing": "wall clock; bag I/O excluded; identical clouds per mode",
            "hardware_note": "local development host, not organizer container",
        },
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "load_ms": load_ms,
            "trees": len(loaded_tree.estimators_),
            "nodes": int(sum(tree.tree_.node_count for tree in loaded_tree.estimators_)),
            "thresholds": thresholds,
            "batch_candidates": len(batch),
            "batch_predict_ms": batch_elapsed * 1000,
            "predict_us_per_candidate": batch_elapsed * 1e6 / len(batch),
            "portable_path": str(portable_path),
            "portable_bytes": portable_path.stat().st_size,
            "portable_load_ms": portable_load_ms,
            "portable_batch_predict_ms": portable_elapsed * 1000,
            "portable_predict_us_per_candidate": portable_elapsed * 1e6 / len(batch),
            "portable_max_abs_error": float(np.max(np.abs(
                portable_scores - loaded_tree.predict_proba(batch)[:, 1]
            ))),
            "portable_hybrid_max_abs_error": hybrid_error,
        },
        "candidates": {
            "mean_per_frame": float(np.mean(candidate_counts)),
            "max_per_frame": int(max(candidate_counts, default=0)),
        },
        "latency": {name: percentiles(values) for name, values in latencies.items()},
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("lidar_geometry/artifacts/extra_trees_candidate.joblib"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("lidar_geometry/artifacts/tree_runtime_benchmark.json"),
    )
    parser.add_argument(
        "--portable-model", type=Path,
        default=Path("lidar_geometry/artifacts/linear_extra_trees_hybrid.npz"),
    )
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args()
    evaluate(
        args.cache, args.bag, args.model, args.portable_model, args.output,
        args.trees,
    )


if __name__ == "__main__":
    main()
