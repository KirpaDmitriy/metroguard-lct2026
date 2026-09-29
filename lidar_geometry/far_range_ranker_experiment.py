import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import SafetyDetection, detect_fast
from lidar_geometry.portable_trees import PortableExtraTrees, PortableFarRangeRanker
from lidar_geometry.risk_model import component_features
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box

DISTANCES_M = (120.0, 150.0)


def far_components(detection: SafetyDetection):
    boundary = detection.reliable_range_max_m or 0.0
    return [
        component
        for component in detection.obstacles
        if component.distance_min_m >= boundary - 1.0
    ]


def training_candidates(root: Path, names: tuple[str, ...], every: int):
    config = DetectorConfig(max_range_m=200.0, path_extrapolation_m=60.0)
    features = []
    labels = []
    normal_frames = []
    positive_events = []
    for bag_index, name in enumerate(names):
        for frame, cloud in selected_frames(bag_path(root, name), every):
            baseline = far_components(detect_fast(cloud, config))
            normal_frames.append([component_features(item) for item in baseline])
            for component in baseline:
                features.append(component_features(component))
                labels.append(0)
            for scenario_index, scenario in enumerate(SCENARIOS):
                for distance in DISTANCES_M:
                    target = replace(scenario.obstacle, distance_m=distance)
                    seed = bag_index * 100_000 + frame * 100 + scenario_index
                    try:
                        injected = inject_box(cloud, target, config, seed)
                    except ValueError:
                        continue
                    if injected.visible_points < 3:
                        continue
                    detection = detect_fast(injected.cloud, config)
                    candidates = far_components(detection)
                    if scenario.expected_alarm:
                        positive_events.append(
                            [
                                component_features(component)
                                for component in candidates
                                if matches_target(component, target)
                            ]
                        )
                    for component in candidates:
                        features.append(component_features(component))
                        labels.append(
                            int(
                                scenario.expected_alarm
                                and matches_target(component, target)
                            )
                        )
    return (
        np.asarray(features),
        np.asarray(labels, dtype=np.int8),
        normal_frames,
        positive_events,
    )


def group_scores(groups: list[list[np.ndarray]], tree: PortableExtraTrees):
    return [tree.score(np.vstack(group)) if group else np.empty(0) for group in groups]


def select_threshold(
    normal_frames: list[list[np.ndarray]],
    positive_events: list[list[np.ndarray]],
    tree: PortableExtraTrees,
) -> float:
    normal_scores = group_scores(normal_frames, tree)
    positive_scores = group_scores(positive_events, tree)
    best = (0.0, 1.0)
    for threshold in np.linspace(0.3, 0.99, 70):
        normal_alarms = sum(np.any(scores >= threshold) for scores in normal_scores)
        recall = sum(np.any(scores >= threshold) for scores in positive_scores) / max(
            1, len(positive_scores)
        )
        if normal_alarms == 0 and recall > best[0]:
            best = (recall, float(threshold))
    return best[1]


def evaluate(
    root: Path,
    names: tuple[str, ...],
    model: PortableFarRangeRanker,
    every: int,
) -> dict:
    config = DetectorConfig(max_range_m=200.0, path_extrapolation_m=60.0)
    normal_frames = normal_alarms = 0
    events = []
    for bag_index, name in enumerate(names):
        for frame, cloud in selected_frames(bag_path(root, name), every):
            baseline = far_components(detect_fast(cloud, config))
            baseline_scores = (
                model.score(np.vstack([component_features(item) for item in baseline]))
                if baseline
                else np.empty(0)
            )
            normal_frames += 1
            normal_alarms += bool(np.any(baseline_scores >= model.threshold))
            for scenario_index, scenario in enumerate(SCENARIOS):
                for distance in DISTANCES_M:
                    target = replace(scenario.obstacle, distance_m=distance)
                    seed = bag_index * 100_000 + frame * 100 + scenario_index
                    try:
                        injected = inject_box(cloud, target, config, seed)
                    except ValueError:
                        continue
                    if injected.visible_points < 3:
                        continue
                    candidates = far_components(detect_fast(injected.cloud, config))
                    matched = [
                        component
                        for component in candidates
                        if matches_target(component, target)
                    ]
                    scores = (
                        model.score(
                            np.vstack([component_features(item) for item in matched])
                        )
                        if matched
                        else np.empty(0)
                    )
                    events.append(
                        {
                            "bag": name,
                            "frame": frame,
                            "scenario": scenario.name,
                            "distance_m": distance,
                            "expected_alarm": scenario.expected_alarm,
                            "visible_points": injected.visible_points,
                            "detected": bool(np.any(scores >= model.threshold)),
                        }
                    )
    positives = [item for item in events if item["expected_alarm"]]
    negatives = [item for item in events if not item["expected_alarm"]]
    return {
        "normal_frames": normal_frames,
        "normal_alarm_frames": normal_alarms,
        "normal_alarm_rate": normal_alarms / max(1, normal_frames),
        "positive_events": len(positives),
        "positive_recall": sum(item["detected"] for item in positives)
        / max(1, len(positives)),
        "negative_events": len(negatives),
        "negative_specificity": sum(not item["detected"] for item in negatives)
        / max(1, len(negatives)),
        "events": events,
    }


def run(root: Path, output_model: Path) -> dict:
    development = ASSUMED_NORMAL[:3]
    test = ASSUMED_NORMAL[3:]
    features, labels, normal_frames, positive_events = training_candidates(
        root, development, every=40
    )
    classifier = ExtraTreesClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ).fit(features, labels)
    tree = PortableExtraTrees.from_sklearn(classifier)
    threshold = select_threshold(normal_frames, positive_events, tree)
    model = PortableFarRangeRanker(tree, threshold)
    model.save(output_model)
    report = {
        "experiment": "EXP-071",
        "protocol": {
            "change": "far-range candidate ranker only",
            "development_bags": list(development),
            "test_bags": list(test),
            "split_unit": "whole organizer bag",
            "distances_m": list(DISTANCES_M),
            "path_extrapolation_m": 60.0,
            "max_range_m": 200.0,
            "threshold_selection": "maximum development event recall with zero development normal-frame alarms",
        },
        "training_candidates": len(labels),
        "training_positives": int(labels.sum()),
        "threshold": threshold,
        "model_bytes": output_model.stat().st_size,
        "test": evaluate(root, test, model, every=40),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("lidar_geometry/artifacts/far_range_extra_trees.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/far_range_ranker_exp071.json"),
    )
    args = parser.parse_args()
    report = run(args.dataset_root, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: report[key] for key in report if key != "protocol"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
