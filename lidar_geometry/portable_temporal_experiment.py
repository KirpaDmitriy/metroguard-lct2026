from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from lidar_geometry.competition_scorecard import (
    ARTIFACTS,
    ROOT,
    fit_folds,
    fixed_decisions,
    fixed_models,
    overlaps_official_bounds,
)
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.evaluate_tracker import target_only
from lidar_geometry.fast_detector import SafetyDetection, detect_fast
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.risk_model import component_features
from lidar_geometry.scenario_catalog import SCENARIOS
from lidar_geometry.synthetic import inject_box
from lidar_geometry.temporal import ComponentTracker, WorldComponentTracker
from lidar_geometry.ego_motion import EgoMotionEstimator


VARIANTS = ("per_frame", "range_3of5", "world_3of5")


def count_episodes(states: list[bool]) -> int:
    return sum(value and (index == 0 or not states[index - 1]) for index, value in enumerate(states))


def selected_detection(geometric: SafetyDetection, items, scores) -> SafetyDetection:
    accepted = tuple(items)
    if accepted:
        nearest = min(accepted, key=lambda item: item.distance_min_m)
        return replace(
            geometric,
            state="OBSTACLE",
            obstacle=True,
            nearest_distance_m=nearest.distance_min_m,
            confidence=max(float(value) for value in scores),
            obstacles=accepted,
            reason="geometry_plus_portable_tree_ranker",
        )
    return replace(
        geometric,
        state="UNKNOWN" if geometric.obstacles else geometric.state,
        obstacle=False,
        nearest_distance_m=None,
        confidence=max((float(value) for value in scores), default=geometric.confidence),
        obstacles=(),
        reason=(
            "geometric_components_rejected_by_portable_tree_ranker"
            if geometric.obstacles else geometric.reason
        ),
    )


def fold_detection(cloud, model) -> SafetyDetection:
    geometric = detect_fast(cloud)
    if not geometric.obstacles:
        return geometric
    features = np.vstack([component_features(item) for item in geometric.obstacles])
    linear_scores = model.linear_predict(features)
    scores = 0.25 * linear_scores + 0.75 * model.tree.predict_proba(features)[:, 1]
    keep = scores >= model.tree_threshold
    return selected_detection(geometric, np.asarray(geometric.obstacles)[keep], scores[keep])


def fixed_detection(cloud, models) -> SafetyDetection:
    geometric = detect_fast(cloud)
    accepted = fixed_decisions(cloud, geometric, None, models)["portable_tree_25_75"]
    tree = models[1]
    if geometric.obstacles:
        scores = tree.score(np.vstack([component_features(item) for item in geometric.obstacles]))
        kept = scores[scores >= tree.threshold]
    else:
        kept = np.empty(0)
    return selected_detection(geometric, accepted, kept)


def trackers():
    return (
        ComponentTracker(required_hits=3, max_missed_frames=4, confirmation_window_frames=5),
        WorldComponentTracker(required_hits=3, max_missed_frames=4, confirmation_window_frames=5),
    )


def variant_states(detection, range_tracker, world_tracker, cumulative_m):
    return {
        "per_frame": detection.obstacle,
        "range_3of5": range_tracker.update(detection).obstacle,
        "world_3of5": world_tracker.update(detection, cumulative_m).obstacle,
    }


def evaluate_normal(root: Path, folds) -> tuple[list[dict], dict]:
    rows = []
    latencies = {name: [] for name in VARIANTS}
    for bag in ASSUMED_NORMAL:
        range_tracker, world_tracker = trackers()
        motion = EgoMotionEstimator()
        states = {name: [] for name in VARIANTS}
        for _, cloud in selected_frames(bag_path(root, bag), 1):
            started = time.perf_counter()
            detection = fold_detection(cloud, folds[bag])
            detected = time.perf_counter()
            range_state = range_tracker.update(detection).obstacle
            ranged = time.perf_counter()
            cumulative_m = motion.update(cloud).cumulative_m
            world_state = world_tracker.update(detection, cumulative_m).obstacle
            worlded = time.perf_counter()
            states["per_frame"].append(detection.obstacle)
            states["range_3of5"].append(range_state)
            states["world_3of5"].append(world_state)
            latencies["per_frame"].append((detected - started) * 1000)
            latencies["range_3of5"].append((ranged - started) * 1000)
            latencies["world_3of5"].append((worlded - started) * 1000)
        rows.append({
            "bag": bag,
            "frames": len(states["per_frame"]),
            "variants": {
                name: {
                    "alarm_frames": sum(values),
                    "alarm_episodes": count_episodes(values),
                }
                for name, values in states.items()
            },
        })
        print(f"portable temporal normal: {bag}", flush=True)
    return rows, {
        name: {
            "p50_ms": float(np.quantile(values, 0.50)),
            "p95_ms": float(np.quantile(values, 0.95)),
        }
        for name, values in latencies.items()
    }


def summarize_normal(rows):
    return {
        name: {
            "frames": sum(row["frames"] for row in rows),
            "alarm_frames": sum(row["variants"][name]["alarm_frames"] for row in rows),
            "alarm_episodes": sum(row["variants"][name]["alarm_episodes"] for row in rows),
        }
        for name in VARIANTS
    }


def evaluate_events(root: Path, folds, frames: int) -> list[dict]:
    rows = []
    positives = [scenario for scenario in SCENARIOS if scenario.expected_alarm]
    for bag_index, bag in enumerate(ASSUMED_NORMAL[:3]):
        clouds = list(selected_frames(bag_path(root, bag), 1, frames))
        for scenario_index, scenario in enumerate(positives):
            range_tracker, world_tracker = trackers()
            states = {name: [] for name in VARIANTS}
            for step, (frame, cloud) in enumerate(clouds):
                target = replace(scenario.obstacle, distance_m=40.0 - step)
                try:
                    injected = inject_box(
                        cloud,
                        target,
                        seed=bag_index * 10_000 + scenario_index * 100 + step,
                    )
                except ValueError:
                    continue
                detection = target_only(fold_detection(injected.cloud, folds[bag]), target)
                decisions = variant_states(detection, range_tracker, world_tracker, float(step))
                if injected.visible_points >= 3:
                    for name, value in decisions.items():
                        states[name].append(value)
            rows.append({
                "bag": bag,
                "scenario": scenario.name,
                "evaluable_frames": len(states["per_frame"]),
                "variants": {
                    name: {"event_detected": any(values), "confirmed_frames": sum(values)}
                    for name, values in states.items()
                },
            })
    return rows


def summarize_events(rows):
    return {
        name: {
            "events": len(rows),
            "detected_events": sum(row["variants"][name]["event_detected"] for row in rows),
            "event_recall": sum(row["variants"][name]["event_detected"] for row in rows) / len(rows),
        }
        for name in VARIANTS
    }


def official_pseudo(bag: Path, motion_path: Path) -> dict:
    motion = json.loads(motion_path.read_text(encoding="utf-8"))["frames_detail"]
    models = fixed_models()
    rows = [{
        "scenario": scenario.name,
        "expected_alarm": scenario.expected_alarm,
        **{name: {"visible_frames": 0, "proposal_frames": 0} for name in VARIANTS},
    } for scenario in SCENARIOS[:10]]
    active = {scenario.name: trackers() for scenario in SCENARIOS[:10]}
    for frame, (_, cloud) in enumerate(iter_bag_messages(bag)):
        detection = fixed_detection(cloud, models)
        cumulative_m = motion[frame]["cumulative_m"]
        for index, scenario in enumerate(SCENARIOS[:10]):
            expected_range = 99.25 + index * 100.0 - cumulative_m
            target = replace(scenario.obstacle, distance_m=expected_range)
            matching = tuple(
                item for item in detection.obstacles
                if abs((item.distance_min_m + item.distance_max_m) / 2 - expected_range) <= 2.0
                and overlaps_official_bounds(item, scenario)
            )
            attributed = selected_detection(detection, matching, [detection.confidence] * len(matching))
            range_tracker, world_tracker = active[scenario.name]
            decisions = variant_states(attributed, range_tracker, world_tracker, cumulative_m)
            if not 3.0 <= expected_range <= 120.0:
                continue
            for name, value in decisions.items():
                rows[index][name]["visible_frames"] += 1
                rows[index][name]["proposal_frames"] += value
    summary = {}
    for name in VARIANTS:
        positive = [row for row in rows if row["expected_alarm"]]
        negative = [row for row in rows if not row["expected_alarm"]]
        summary[name] = {
            "positive_scenarios_with_proposal": sum(row[name]["proposal_frames"] > 0 for row in positive),
            "positive_scenarios": len(positive),
            "positive_coverage": sum(row[name]["proposal_frames"] > 0 for row in positive) / len(positive),
            "negative_scenarios_without_proposal": sum(row[name]["proposal_frames"] == 0 for row in negative),
            "negative_scenarios": len(negative),
            "negative_specificity": sum(row[name]["proposal_frames"] == 0 for row in negative) / len(negative),
        }
    return {"models": summary, "scenarios": rows}


def reduction(candidate: dict, baseline: dict, metric: str) -> float:
    return (baseline[metric] - candidate[metric]) / baseline[metric] if baseline[metric] else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--synthetic-frames", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS / "portable_temporal_exp068.json",
    )
    args = parser.parse_args()
    folds, fold_protocol = fit_folds(ARTIFACTS / "spatial_patch_cache_v1.npz")
    normal_rows, runtime = evaluate_normal(args.dataset_root, folds)
    normal = summarize_normal(normal_rows)
    event_rows = evaluate_events(args.dataset_root, folds, args.synthetic_frames)
    events = summarize_events(event_rows)
    official = official_pseudo(
        ROOT / "external_data/hackathon/synthetic_official/data/cloud_with_fake_obj/cloud_with_fake_obj_0.db3",
        ARTIFACTS / "official_ego_motion.json",
    )
    comparisons = {
        name: {
            "alarm_frame_reduction": reduction(normal[name], normal["per_frame"], "alarm_frames"),
            "alarm_episode_reduction": reduction(normal[name], normal["per_frame"], "alarm_episodes"),
            "events_lost": events["per_frame"]["detected_events"] - events[name]["detected_events"],
        }
        for name in VARIANTS[1:]
    }
    report = {
        "experiment": "EXP-068",
        "protocol": {
            "normal_split": fold_protocol,
            "normal_frames": "all frames from five whole organizer bags",
            "event_sequences": "same 21 fixed exact-mask sequences used by EXP-065",
            "official_ground_truth": False,
            "official_warning": "forensic coverage from approximate organizer order and spacing",
            "frozen": ["geometry", "features", "portable model family", "25/75 weights", "thresholds"],
        },
        "normal_summary": normal,
        "runtime": runtime,
        "event_summary": events,
        "official_pseudo": official,
        "comparisons": comparisons,
        "normal_by_bag": normal_rows,
        "event_sequences": event_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "normal_summary": normal,
        "runtime": runtime,
        "event_summary": events,
        "official_pseudo": official["models"],
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
