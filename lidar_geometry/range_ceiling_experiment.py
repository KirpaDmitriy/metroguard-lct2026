import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.runtime_detector import ALGORITHMS, RuntimeDetector
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box

CEILINGS_M = (150.0, 200.0, 300.0)
DISTANCES_M = (80.0, 100.0, 120.0, 150.0, 180.0)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def summarize(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key.split("+"))].append(row)
    result = []
    for group, items in sorted(groups.items()):
        values = group if isinstance(group, tuple) else (group,)
        entry = dict(zip(key.split("+"), values))
        evaluable = [item for item in items if item["evaluable"]]
        positives = [item for item in evaluable if item["expected_alarm"]]
        negatives = [item for item in evaluable if not item["expected_alarm"]]
        entry.update(
            evaluable=len(evaluable),
            positive_recall=(
                sum(item["matched"] for item in positives) / len(positives)
                if positives
                else None
            ),
            negative_specificity=(
                sum(not item["matched"] for item in negatives) / len(negatives)
                if negatives
                else None
            ),
        )
        result.append(entry)
    return result


def run(root: Path, frames_per_bag: int, sample_every: int) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    generation_config = DetectorConfig(max_range_m=max(CEILINGS_M))
    detectors = {
        (algorithm, ceiling): RuntimeDetector.from_root(
            algorithm,
            project_root,
            config=DetectorConfig(max_range_m=ceiling),
        )
        for algorithm in ALGORITHMS
        for ceiling in CEILINGS_M
    }
    injected_rows = []
    baseline = defaultdict(lambda: {"frames": 0, "alarms": 0, "latency_ms": []})
    reliable_ranges = []
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        frames = list(
            selected_frames(bag_path(root, name), sample_every, frames_per_bag)
        )
        for frame, cloud in frames:
            for key, detector in detectors.items():
                started = time.perf_counter()
                result = detector(cloud)
                elapsed = (time.perf_counter() - started) * 1000
                baseline[key]["frames"] += 1
                baseline[key]["alarms"] += result.state == "OBSTACLE"
                baseline[key]["latency_ms"].append(elapsed)
                if key[0] == "geometry" and key[1] == max(CEILINGS_M):
                    reliable_ranges.append(result.reliable_range_max_m or 0.0)
            for scenario_index, scenario in enumerate(SCENARIOS):
                for distance in DISTANCES_M:
                    target = replace(scenario.obstacle, distance_m=distance)
                    seed = bag_index * 100_000 + frame * 100 + scenario_index
                    try:
                        injected = inject_box(cloud, target, generation_config, seed)
                    except ValueError:
                        injected = None
                    visible = injected.visible_points if injected is not None else 0
                    for (algorithm, ceiling), detector in detectors.items():
                        matched = False
                        state = "NOT_EVALUABLE"
                        if injected is not None and visible >= 3:
                            detection = detector(injected.cloud)
                            state = detection.state
                            matched = any(
                                matches_target(component, target)
                                for component in detection.obstacles
                            )
                        injected_rows.append(
                            {
                                "bag": name,
                                "frame": frame,
                                "scenario": scenario.name,
                                "expected_alarm": scenario.expected_alarm,
                                "distance_m": distance,
                                "visible_points": visible,
                                "evaluable": visible >= 3,
                                "algorithm": algorithm,
                                "ceiling_m": ceiling,
                                "matched": matched,
                                "state": state,
                            }
                        )
    baseline_rows = []
    for (algorithm, ceiling), values in sorted(baseline.items()):
        baseline_rows.append(
            {
                "algorithm": algorithm,
                "ceiling_m": ceiling,
                "frames": values["frames"],
                "alarm_rate": values["alarms"] / values["frames"],
                "latency_p50_ms": statistics.median(values["latency_ms"]),
                "latency_p95_ms": percentile(values["latency_ms"], 0.95),
            }
        )
    return {
        "experiment": "EXP-069",
        "protocol": {
            "change": "max_range_m only",
            "ceilings_m": list(CEILINGS_M),
            "distances_m": list(DISTANCES_M),
            "algorithms": list(ALGORITHMS),
            "backgrounds": list(ASSUMED_NORMAL),
            "frames_per_bag": frames_per_bag,
            "sample_every": sample_every,
            "minimum_visible_points": 3,
            "target_attribution": "exact overlap with injected AABB",
        },
        "reliable_track_range_m": {
            "minimum": min(reliable_ranges),
            "median": statistics.median(reliable_ranges),
            "maximum": max(reliable_ranges),
        },
        "baseline": baseline_rows,
        "by_distance": summarize(injected_rows, "algorithm+ceiling_m+distance_m"),
        "rows": injected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--frames-per-bag", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/range_ceiling_exp069.json"),
    )
    args = parser.parse_args()
    report = run(args.dataset_root, args.frames_per_bag, args.sample_every)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("reliable_track_range_m", "baseline", "by_distance")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
