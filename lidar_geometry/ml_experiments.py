"""Compare a supervised component classifier with one-class anomaly detection.

The script intentionally uses only the Python standard library.  It consumes
the high-recall geometric proposals from :mod:`detect_obstacles`; it does not
try to learn directly from hundreds of thousands of raw lidar points.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import statistics

from lidar_geometry.detect_obstacles import DetectorConfig, detect
from lidar_geometry.pointcloud2 import iter_bag_messages


NORMAL_BAGS = (
    "roundT_pressureGate_roundT",
    "roundT_doubleT",
    "roundT_squareT_pressureGate_squareT",
    "squareT_platform_squareT_switch",
    "doubleT_platform",
)
OBSTACLE_BAG = "doubleT_obstacle"

FEATURES = (
    "log_points",
    "log_voxels",
    "points_per_voxel",
    "distance_span_m",
    "lateral_span_m",
    "height_span_m",
    "height_min_m",
    "height_max_m",
    "surface_residual_max_m",
    "surface_residual_mean_m",
    "abs_lateral_center_m",
    "side_clearance_m",
    "log_intensity",
    "range_fraction",
    "log_frame_candidates",
)


def proposal_config() -> DetectorConfig:
    """Relax only component filters; retain rail and surface estimation."""
    return replace(
        DetectorConfig(),
        min_cluster_points=3,
        min_cluster_voxels=1,
        min_cluster_vertical_extent_m=0.0,
        min_cluster_top_height_m=-0.05,
        min_cluster_horizontal_extent_m=0.0,
        rail_like_max_height_m=-1.0,
    )


def component_record(bag: str, frame: int, detection, obstacle) -> dict[str, float | int | str]:
    record: dict[str, float | int | str] = {
        "bag": bag,
        "frame": frame,
        **asdict(obstacle),
        "frame_candidates": detection.candidates,
        "track_bins": detection.track_bins,
        "reliable_range_min_m": detection.reliable_range_min_m or 0.0,
        "reliable_range_max_m": detection.reliable_range_max_m or 0.0,
    }
    return record


def collect(dataset_root: Path, every: int, output: Path) -> list[dict]:
    config = proposal_config()
    records: list[dict] = []
    for bag in (*NORMAL_BAGS, OBSTACLE_BAG):
        path = dataset_root / bag / f"{bag}_0.db3"
        if not path.exists():
            raise FileNotFoundError(path)
        sampled = 0
        for frame, (_timestamp, cloud) in enumerate(iter_bag_messages(path)):
            if frame % every:
                continue
            result = detect(cloud, config)
            records.extend(component_record(bag, frame, result, item) for item in result.obstacles)
            sampled += 1
        print(json.dumps({"collected": bag, "frames": sampled, "records": len(records)}))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    return records


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def feature_map(record: dict) -> dict[str, float]:
    distance_span = max(0.0, record["distance_max_m"] - record["distance_min_m"])
    lateral_span = max(0.0, record["lateral_max_m"] - record["lateral_min_m"])
    height_span = max(0.0, record["height_max_m"] - record["height_min_m"])
    lateral_center = (record["lateral_min_m"] + record["lateral_max_m"]) / 2
    range_min = record["reliable_range_min_m"]
    range_max = record["reliable_range_max_m"]
    range_fraction = (record["distance_min_m"] - range_min) / max(1.0, range_max - range_min)
    return {
        "log_points": math.log1p(record["points"]),
        "log_voxels": math.log1p(record["voxels"]),
        "points_per_voxel": record["points"] / max(1, record["voxels"]),
        "distance_span_m": distance_span,
        "lateral_span_m": lateral_span,
        "height_span_m": height_span,
        "height_min_m": record["height_min_m"],
        "height_max_m": record["height_max_m"],
        "surface_residual_max_m": record["surface_residual_max_m"],
        "surface_residual_mean_m": record["surface_residual_mean_m"],
        "abs_lateral_center_m": abs(lateral_center),
        "side_clearance_m": 1.05 - abs(lateral_center) - lateral_span / 2,
        "log_intensity": math.log1p(max(0.0, record["intensity_mean"])),
        "range_fraction": range_fraction,
        "log_frame_candidates": math.log1p(record["frame_candidates"]),
    }


def vector(record: dict) -> list[float]:
    values = feature_map(record)
    return [values[name] for name in FEATURES]


class Standardizer:
    def fit(self, rows: list[list[float]]) -> None:
        columns = list(zip(*rows))
        self.center = [statistics.median(column) for column in columns]
        self.scale = []
        for column, center in zip(columns, self.center):
            mad = statistics.median(abs(value - center) for value in column)
            self.scale.append(max(1e-6, 1.4826 * mad))

    def transform(self, row: list[float]) -> list[float]:
        return [(value - center) / scale for value, center, scale in zip(row, self.center, self.scale)]


class LogisticClassifier:
    def fit(self, rows: list[list[float]], labels: list[int], iterations: int = 350) -> None:
        self.standardizer = Standardizer()
        self.standardizer.fit(rows)
        x = [self.standardizer.transform(row) for row in rows]
        self.weights = [0.0] * (len(FEATURES) + 1)
        for step in range(iterations):
            gradient = [0.0] * len(self.weights)
            for row, label in zip(x, labels):
                score = self.weights[0] + sum(w * value for w, value in zip(self.weights[1:], row))
                score = max(-30.0, min(30.0, score))
                error = 1.0 / (1.0 + math.exp(-score)) - label
                gradient[0] += error
                for index, value in enumerate(row, 1):
                    gradient[index] += error * value
            rate = 0.08 / math.sqrt(1 + step / 100)
            for index in range(len(self.weights)):
                regularization = 0.002 * self.weights[index] if index else 0.0
                self.weights[index] -= rate * (gradient[index] / len(x) + regularization)

    def score(self, row: list[float]) -> float:
        row = self.standardizer.transform(row)
        value = self.weights[0] + sum(w * item for w, item in zip(self.weights[1:], row))
        value = max(-30.0, min(30.0, value))
        return 1.0 / (1.0 + math.exp(-value))


class KnnAnomalyDetector:
    def fit(self, rows: list[list[float]], k: int = 5, max_prototypes: int = 600) -> None:
        self.standardizer = Standardizer()
        self.standardizer.fit(rows)
        # Exact all-pairs kNN is quadratic. A fixed-seed prototype subset keeps
        # this dependency-free experiment fast and reproducible.
        if len(rows) > max_prototypes:
            rows = random.Random(41).sample(rows, max_prototypes)
        self.rows = [self.standardizer.transform(row) for row in rows]
        self.k = min(k, len(self.rows))

    def score(self, row: list[float]) -> float:
        query = self.standardizer.transform(row)
        distances = sorted(
            sum((a - b) ** 2 for a, b in zip(query, reference))
            for reference in self.rows
        )
        return sum(distances[: self.k]) / self.k


def synthetic_positives(negatives: list[dict], count: int, seed: int = 17) -> list[dict]:
    rng = random.Random(seed)
    generated = []
    for _ in range(count):
        base = dict(rng.choice(negatives))
        width = rng.uniform(0.25, 1.2)
        length = rng.uniform(0.25, 1.8)
        height = rng.uniform(0.10, 2.2)
        bottom = rng.uniform(0.0, min(0.25, max(0.0, 3.0 - height)))
        center = rng.uniform(-max(0.0, 0.85 - width / 2), max(0.0, 0.85 - width / 2))
        distance = rng.uniform(
            max(5.0, base["reliable_range_min_m"]),
            max(6.0, base["reliable_range_max_m"] - length),
        )
        projected_area = max(width * height, width * length)
        points = max(8, round(rng.uniform(80, 500) * projected_area / (1 + distance / 60)))
        base.update(
            points=points,
            voxels=max(2, round(points / rng.uniform(2.0, 7.0))),
            distance_min_m=distance,
            distance_max_m=distance + length,
            lateral_min_m=center - width / 2,
            lateral_max_m=center + width / 2,
            height_min_m=bottom,
            height_max_m=bottom + height,
            surface_residual_max_m=height,
            surface_residual_mean_m=rng.uniform(0.06, max(0.061, height * 0.7)),
            intensity_mean=math.exp(rng.uniform(0.0, 4.2)) - 1,
        )
        generated.append(base)
    return generated


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def frame_max_scores(records: list[dict], scores: list[float]) -> list[float]:
    maxima: dict[tuple[str, int], float] = {}
    for item, score in zip(records, scores):
        key = (item["bag"], item["frame"])
        maxima[key] = max(score, maxima.get(key, -math.inf))
    return list(maxima.values())


def evaluate(records: list[dict], output: Path) -> dict:
    normal = [item for item in records if item["bag"] in NORMAL_BAGS]
    obstacle = [item for item in records if item["bag"] == OBSTACLE_BAG]
    if not normal or not obstacle:
        raise ValueError("Both normal and obstacle-bag proposals are required")

    folds = []
    for held_out in NORMAL_BAGS:
        train = [item for item in normal if item["bag"] != held_out]
        test = [item for item in normal if item["bag"] == held_out]
        if not train or not test:
            continue
        synthetic = synthetic_positives(train, max(500, len(train)), seed=17)
        supervised = LogisticClassifier()
        supervised.fit(
            [vector(item) for item in train + synthetic],
            [0] * len(train) + [1] * len(synthetic),
        )
        anomaly = KnnAnomalyDetector()
        anomaly.fit([vector(item) for item in train])
        supervised_train_scores = [supervised.score(vector(item)) for item in train]
        anomaly_train_scores = [anomaly.score(vector(item)) for item in train]
        supervised_train_frame_scores = frame_max_scores(train, supervised_train_scores)
        anomaly_train_frame_scores = frame_max_scores(train, anomaly_train_scores)
        supervised_threshold = percentile(supervised_train_frame_scores, 0.99)
        anomaly_threshold = percentile(anomaly_train_frame_scores, 0.99)
        supervised_test_scores = [supervised.score(vector(item)) for item in test]
        anomaly_test_scores = [anomaly.score(vector(item)) for item in test]
        candidate_frames = sorted({item["frame"] for item in test})
        folds.append({
            "held_out": held_out,
            "normal_components": len(test),
            "supervised_false_positive_rate": (
                sum(score > supervised_threshold for score in supervised_test_scores) / len(test)
            ),
            "anomaly_false_positive_rate": (
                sum(score > anomaly_threshold for score in anomaly_test_scores) / len(test)
            ),
            "candidate_bearing_frame_fpr": {
                "supervised": len({
                    item["frame"] for item, score in zip(test, supervised_test_scores)
                    if score > supervised_threshold
                }) / len(candidate_frames),
                "anomaly": len({
                    item["frame"] for item, score in zip(test, anomaly_test_scores)
                    if score > anomaly_threshold
                }) / len(candidate_frames),
            },
            "frame_fpr_at_p95_threshold": {
                "supervised": len({
                    item["frame"] for item, score in zip(test, supervised_test_scores)
                    if score > percentile(supervised_train_frame_scores, 0.95)
                }) / len(candidate_frames),
                "anomaly": len({
                    item["frame"] for item, score in zip(test, anomaly_test_scores)
                    if score > percentile(anomaly_train_frame_scores, 0.95)
                }) / len(candidate_frames),
            },
        })

    synthetic = synthetic_positives(normal, max(1000, len(normal)), seed=23)
    supervised = LogisticClassifier()
    supervised.fit(
        [vector(item) for item in normal + synthetic],
        [0] * len(normal) + [1] * len(synthetic),
    )
    anomaly = KnnAnomalyDetector()
    anomaly.fit([vector(item) for item in normal])
    supervised_normal_scores = [supervised.score(vector(item)) for item in normal]
    anomaly_normal_scores = [anomaly.score(vector(item)) for item in normal]
    supervised_normal_frame_scores = frame_max_scores(normal, supervised_normal_scores)
    anomaly_normal_frame_scores = frame_max_scores(normal, anomaly_normal_scores)
    supervised_threshold = percentile(supervised_normal_frame_scores, 0.99)
    anomaly_threshold = percentile(anomaly_normal_frame_scores, 0.99)

    ranked = []
    for item in obstacle:
        row = dict(item)
        row["supervised_score"] = supervised.score(vector(item))
        row["anomaly_score"] = anomaly.score(vector(item))
        row["supervised_flag"] = row["supervised_score"] > supervised_threshold
        row["anomaly_flag"] = row["anomaly_score"] > anomaly_threshold
        ranked.append(row)
    ranked.sort(key=lambda item: max(
        item["supervised_score"] / max(supervised_threshold, 1e-9),
        item["anomaly_score"] / max(anomaly_threshold, 1e-9),
    ), reverse=True)

    operating_points = {}
    for fraction in (0.90, 0.95, 0.99):
        supervised_cut = percentile(supervised_normal_frame_scores, fraction)
        anomaly_cut = percentile(anomaly_normal_frame_scores, fraction)
        operating_points[f"p{round(fraction * 100)}"] = {
            "supervised_threshold": supervised_cut,
            "anomaly_threshold": anomaly_cut,
            "obstacle_bag_flagged_frames": {
                "supervised": sorted({
                    item["frame"] for item in ranked if item["supervised_score"] > supervised_cut
                }),
                "anomaly": sorted({
                    item["frame"] for item in ranked if item["anomaly_score"] > anomaly_cut
                }),
            },
        }

    result = {
        "protocol": "leave-one-normal-sequence-out; thresholds are p99 of train-frame maximum scores",
        "normal_components": len(normal),
        "obstacle_bag_components": len(obstacle),
        "folds": folds,
        "mean_normal_fpr": {
            "supervised": statistics.mean(item["supervised_false_positive_rate"] for item in folds),
            "anomaly": statistics.mean(item["anomaly_false_positive_rate"] for item in folds),
        },
        "thresholds": {"supervised": supervised_threshold, "anomaly": anomaly_threshold},
        "operating_points": operating_points,
        "supervised_standardized_coefficients": sorted(
            ({"feature": name, "weight": weight} for name, weight in zip(FEATURES, supervised.weights[1:])),
            key=lambda item: abs(item["weight"]),
            reverse=True,
        ),
        "obstacle_bag_flagged_components": {
            "supervised": sum(item["supervised_flag"] for item in ranked),
            "anomaly": sum(item["anomaly_flag"] for item in ranked),
            "both": sum(item["supervised_flag"] and item["anomaly_flag"] for item in ranked),
        },
        "obstacle_bag_flagged_frames": {
            "supervised": sorted({item["frame"] for item in ranked if item["supervised_flag"]}),
            "anomaly": sorted({item["frame"] for item in ranked if item["anomaly_flag"]}),
            "both": sorted({
                item["frame"] for item in ranked
                if item["supervised_flag"] and item["anomaly_flag"]
            }),
        },
        "top_obstacle_bag_candidates": ranked[:20],
        "warning": "Obstacle bag has no component labels; flagged counts are not recall.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--cache", type=Path, default=Path("lidar_geometry/artifacts/components.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("lidar_geometry/artifacts/ml_comparison.json"))
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    records = load_records(args.cache) if args.reuse_cache else collect(
        args.dataset_root, args.every, args.cache
    )
    result = evaluate(records, args.report)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
