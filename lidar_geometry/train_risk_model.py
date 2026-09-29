"""Train and audit the compact infrastructure-vs-object proposal ranker."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.risk_model import FEATURE_NAMES, component_features, fit_logistic
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box


def collect(root: Path, every: int, synthetic_every: int):
    rows, labels, groups, records = [], [], [], []
    config = DetectorConfig()
    for bag_index, name in enumerate(ASSUMED_NORMAL):
        path = bag_path(root, name)
        for frame, cloud in selected_frames(path, every):
            result = detect_fast(cloud, config)
            for item in result.obstacles:
                rows.append(component_features(item)); labels.append(0); groups.append(name)
                records.append({"bag": name, "frame": frame, "label": 0, "source": "real"})
            if frame % synthetic_every:
                continue
            for scenario_index, scenario in enumerate(SCENARIOS):
                for distance in (20.0, 40.0, 60.0):
                    obstacle = replace(scenario.obstacle, distance_m=distance)
                    try:
                        injected = inject_box(
                            cloud,
                            obstacle,
                            config,
                            seed=bag_index * 100_000 + frame * 100 + scenario_index,
                        )
                    except ValueError:
                        continue
                    if injected.visible_points < 3:
                        continue
                    detected = detect_fast(injected.cloud, config)
                    matches = [
                        item for item in detected.obstacles
                        if matches_target(item, obstacle)
                    ]
                    if not matches:
                        continue
                    item = max(matches, key=lambda value: value.points)
                    label = int(scenario.expected_alarm)
                    rows.append(component_features(item)); labels.append(label); groups.append(name)
                    records.append({
                        "bag": name, "frame": frame, "label": label,
                        "source": "synthetic",
                        "scenario": scenario.name, "distance_m": distance,
                        "visible_points": injected.visible_points,
                    })
    return np.vstack(rows), np.asarray(labels), np.asarray(groups), records


def scores(model, x):
    return np.asarray([
        1.0 / (1.0 + np.exp(-np.clip(
            ((row - model.mean) / model.scale) @ model.weights + model.bias,
            -30,
            30,
        )))
        for row in x
    ])


def calibrate_threshold(model, x, y, records, quantile: float = 0.99):
    values = scores(model, x)
    frame_maxima = {}
    for score, label, record in zip(values, y, records):
        if label or record["source"] != "real":
            continue
        key = (record["bag"], record["frame"])
        frame_maxima[key] = max(score, frame_maxima.get(key, 0.0))
    threshold = float(np.quantile(list(frame_maxima.values()), quantile, method="higher"))
    return replace(model, threshold=threshold)


def rates(model, x, y, records):
    values = scores(model, x)
    predicted = values >= model.threshold
    real_negative_frames = {}
    for prediction, label, record in zip(predicted, y, records):
        if label or record["source"] != "real":
            continue
        key = (record["bag"], record["frame"])
        real_negative_frames[key] = prediction or real_negative_frames.get(key, False)
    synthetic_negative = np.asarray([
        prediction for prediction, label, record in zip(predicted, y, records)
        if not label and record["source"] == "synthetic"
    ])
    return {
        "samples": len(y), "positive_samples": int(y.sum()),
        "recall": float(predicted[y == 1].mean()) if np.any(y == 1) else None,
        "false_positive_rate": float(predicted[y == 0].mean()) if np.any(y == 0) else None,
        "real_frame_false_positive_rate": (
            sum(real_negative_frames.values()) / len(real_negative_frames)
            if real_negative_frames else None
        ),
        "synthetic_negative_false_positive_rate": (
            float(synthetic_negative.mean()) if len(synthetic_negative) else None
        ),
        "score_median_positive": float(np.median(values[y == 1])) if np.any(y == 1) else None,
        "score_p95_negative": float(np.quantile(values[y == 0], .95)) if np.any(y == 0) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--synthetic-every", type=int, default=50)
    parser.add_argument("--invariant", action="store_true")
    parser.add_argument("--critical-weight", type=int, default=1)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--model", type=Path, default=Path("lidar_geometry/models/risk_model.json"))
    parser.add_argument("--report", type=Path, default=Path("lidar_geometry/artifacts/risk_model_audit.json"))
    args = parser.parse_args()
    if args.cache and args.cache.exists():
        cached = np.load(args.cache, allow_pickle=False)
        x = cached["x"]
        y = cached["y"]
        groups = cached["groups"]
        records = json.loads(str(cached["records_json"]))
        if x.shape[1] == len(FEATURE_NAMES) - 1:
            lateral = x[:, FEATURE_NAMES.index("lateral_span_m")]
            center = x[:, FEATURE_NAMES.index("abs_lateral_center_m")]
            x = np.column_stack((x, 1.05 - center - lateral / 2))
    else:
        x, y, groups, records = collect(args.dataset_root, args.every, args.synthetic_every)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.cache,
                x=x,
                y=y,
                groups=groups,
                records_json=json.dumps(records, ensure_ascii=False),
            )
    critical_scenarios = {"minimum_on_rail", "minimum_edge_inside", "hanging_cable"}
    critical = [
        index for index, record in enumerate(records)
        if record.get("scenario") in critical_scenarios and record["label"] == 1
    ]
    if args.critical_weight > 1 and critical:
        repeats = args.critical_weight - 1
        x = np.concatenate([x, np.repeat(x[critical], repeats, axis=0)])
        y = np.concatenate([y, np.repeat(y[critical], repeats, axis=0)])
        groups = np.concatenate([groups, np.repeat(groups[critical], repeats, axis=0)])
        records.extend(dict(records[index]) for index in critical for _ in range(repeats))
    ignored_features = ()
    if args.invariant:
        ignored_features = ("abs_lateral_center_m", "height_min_m", "height_max_m")
        for name in ignored_features:
            x[:, FEATURE_NAMES.index(name)] = 0.0
    folds = []
    for held_out in ASSUMED_NORMAL:
        train = groups != held_out
        test = ~train
        model = fit_logistic(x[train], y[train])
        train_records = [record for record, keep in zip(records, train) if keep]
        test_records = [record for record, keep in zip(records, test) if keep]
        model = calibrate_threshold(model, x[train], y[train], train_records)
        folds.append({
            "held_out_bag": held_out,
            "threshold": model.threshold,
            **rates(model, x[test], y[test], test_records),
        })
    model = fit_logistic(x, y)
    model = calibrate_threshold(model, x, y, records)
    model.save(args.model)
    audit = {
        "protocol": {
            "negative_label": "geometric components in filenames without obstacle token (weak)",
            "positive_label": "ray-consistent AABB inserted into the same real frames",
            "split": "leave-one-recording-out",
            "feature_names": FEATURE_NAMES,
            "ignored_absolute_position_features": ignored_features,
            "critical_scenario_weight": args.critical_weight,
            "threshold": model.threshold,
        },
        "all_data": rates(model, x, y, records),
        "leave_one_recording_out": folds,
        "weights": dict(zip(FEATURE_NAMES, model.weights)),
        "collection_records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.model), "report": str(args.report),
                      "all_data": audit["all_data"], "folds": folds}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
