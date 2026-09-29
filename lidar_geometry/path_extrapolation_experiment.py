import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.runtime_detector import ALGORITHMS, RuntimeDetector
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box

DISTANCES_M = (120.0, 150.0, 180.0)
EXTRAPOLATIONS_M = (0.0, 60.0)


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["algorithm"], row["extrapolation_m"], row["distance_m"])].append(
            row
        )
    summary = []
    for (algorithm, extrapolation, distance), items in sorted(groups.items()):
        positives = [item for item in items if item["expected_alarm"]]
        negatives = [item for item in items if not item["expected_alarm"]]
        summary.append(
            {
                "algorithm": algorithm,
                "extrapolation_m": extrapolation,
                "distance_m": distance,
                "evaluable_positives": len(positives),
                "positive_recall": (
                    sum(item["matched"] for item in positives) / len(positives)
                    if positives
                    else None
                ),
                "negative_specificity": (
                    sum(not item["matched"] for item in negatives) / len(negatives)
                    if negatives
                    else None
                ),
            }
        )
    return summary


def run(root: Path, frames_per_bag: int, sample_every: int) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    detectors = {
        (algorithm, extrapolation): RuntimeDetector.from_root(
            algorithm,
            project_root,
            config=DetectorConfig(
                max_range_m=200.0,
                path_extrapolation_m=extrapolation,
            ),
        )
        for algorithm in ALGORITHMS
        for extrapolation in EXTRAPOLATIONS_M
    }
    generation_config = DetectorConfig(
        max_range_m=200.0,
        path_extrapolation_m=max(EXTRAPOLATIONS_M),
    )
    rows = []
    background = defaultdict(lambda: {"frames": 0, "alarms": 0})
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        for frame, cloud in selected_frames(
            bag_path(root, name), sample_every, frames_per_bag
        ):
            for key, detector in detectors.items():
                result = detector(cloud)
                background[key]["frames"] += 1
                background[key]["alarms"] += result.state == "OBSTACLE"
            for scenario_index, scenario in enumerate(SCENARIOS):
                for distance in DISTANCES_M:
                    target = replace(scenario.obstacle, distance_m=distance)
                    seed = bag_index * 100_000 + frame * 100 + scenario_index
                    try:
                        injected = inject_box(cloud, target, generation_config, seed)
                    except ValueError:
                        continue
                    if injected.visible_points < 3:
                        continue
                    for (algorithm, extrapolation), detector in detectors.items():
                        result = detector(injected.cloud)
                        rows.append(
                            {
                                "bag": name,
                                "frame": frame,
                                "scenario": scenario.name,
                                "expected_alarm": scenario.expected_alarm,
                                "distance_m": distance,
                                "visible_points": injected.visible_points,
                                "algorithm": algorithm,
                                "extrapolation_m": extrapolation,
                                "matched": any(
                                    matches_target(component, target)
                                    for component in result.obstacles
                                ),
                            }
                        )
    background_summary = []
    for (algorithm, extrapolation), values in sorted(background.items()):
        background_summary.append(
            {
                "algorithm": algorithm,
                "extrapolation_m": extrapolation,
                "frames": values["frames"],
                "alarm_rate": values["alarms"] / values["frames"],
            }
        )
    return {
        "experiment": "EXP-070",
        "protocol": {
            "change": "bounded median-tangent path extrapolation only",
            "max_range_m": 200.0,
            "extrapolations_m": list(EXTRAPOLATIONS_M),
            "distances_m": list(DISTANCES_M),
            "algorithms": list(ALGORITHMS),
            "backgrounds": list(ASSUMED_NORMAL),
            "frames_per_bag": frames_per_bag,
            "sample_every": sample_every,
            "minimum_visible_points": 3,
            "target_attribution": "exact overlap with injected AABB",
        },
        "background": background_summary,
        "summary": aggregate(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--frames-per-bag", type=int, default=3)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/path_extrapolation_exp070.json"),
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
            {key: report[key] for key in ("background", "summary")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
