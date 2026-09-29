import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from lidar_geometry.competition_scorecard import ARTIFACTS, fit_folds
from lidar_geometry.ego_motion import EgoMotionEstimator
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.evaluate_tracker import target_only
from lidar_geometry.evidence_memory import RangeEvidenceMemory, WorldEvidenceMemory
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.online_memory_experiment import episodes, percentile
from lidar_geometry.risk_model import component_features
from lidar_geometry.scenario_catalog import SCENARIOS
from lidar_geometry.synthetic import inject_box


def scores(geometric, model) -> np.ndarray:
    if not geometric.obstacles:
        return np.empty(0)
    features = np.vstack([component_features(item) for item in geometric.obstacles])
    linear = model.linear_predict(features)
    tree = model.tree.predict_proba(features)[:, 1]
    return 0.25 * linear + 0.75 * tree


def memory(model, association: str):
    memory_type = WorldEvidenceMemory if association == "world" else RangeEvidenceMemory
    return memory_type(model.tree_threshold, score_floor_ratio=0.75)


def evaluate_normal(
    root: Path,
    folds,
    association: str,
) -> tuple[list[dict], list[float]]:
    rows = []
    latencies = []
    for bag in ASSUMED_NORMAL:
        model = folds[bag]
        evidence = memory(model, association)
        motion = EgoMotionEstimator() if association == "world" else None
        per_frame = []
        remembered = []
        for _, cloud in selected_frames(bag_path(root, bag), 1):
            started = time.perf_counter()
            geometric = detect_fast(cloud)
            candidate_scores = scores(geometric, model)
            cumulative_m = motion.update(cloud).cumulative_m if motion else 0.0
            decision = evidence.update(
                geometric,
                candidate_scores,
                cumulative_m,
            )
            latencies.append((time.perf_counter() - started) * 1000)
            per_frame.append(bool(np.any(candidate_scores >= model.tree_threshold)))
            remembered.append(decision.obstacle)
        rows.append(
            {
                "bag": bag,
                "frames": len(per_frame),
                "per_frame_alarm_frames": sum(per_frame),
                "per_frame_alarm_episodes": episodes(per_frame),
                "memory_alarm_frames": sum(remembered),
                "memory_alarm_episodes": episodes(remembered),
            }
        )
        print(f"evidence memory normal: {bag}", flush=True)
    return rows, latencies


def evaluate_events(
    root: Path,
    folds,
    frame_count: int,
    association: str,
) -> list[dict]:
    rows = []
    positives = [scenario for scenario in SCENARIOS if scenario.expected_alarm]
    for bag_index, bag in enumerate(ASSUMED_NORMAL[:3]):
        model = folds[bag]
        clouds = list(selected_frames(bag_path(root, bag), 1, frame_count))
        for scenario_index, scenario in enumerate(positives):
            evidence = memory(model, association)
            per_frame = []
            remembered = []
            for step, (_, cloud) in enumerate(clouds):
                target = replace(scenario.obstacle, distance_m=40.0 - step)
                try:
                    injected = inject_box(
                        cloud,
                        target,
                        seed=bag_index * 10_000 + scenario_index * 100 + step,
                    )
                except ValueError:
                    continue
                geometric = target_only(detect_fast(injected.cloud), target)
                candidate_scores = scores(geometric, model)
                cumulative_m = float(step) if association == "world" else 0.0
                decision = evidence.update(geometric, candidate_scores, cumulative_m)
                if injected.visible_points < 3:
                    continue
                per_frame.append(bool(np.any(candidate_scores >= model.tree_threshold)))
                remembered.append(decision.obstacle)
            rows.append(
                {
                    "bag": bag,
                    "scenario": scenario.name,
                    "evaluable_frames": len(per_frame),
                    "per_frame_detected": any(per_frame),
                    "memory_detected": any(remembered),
                    "per_frame_confirmed_frames": sum(per_frame),
                    "memory_confirmed_frames": sum(remembered),
                }
            )
    return rows


def run(
    root: Path,
    cache: Path,
    frame_count: int,
    association: str,
    experiment: str,
) -> dict:
    folds, fold_protocol = fit_folds(cache)
    normal, latencies = evaluate_normal(root, folds, association)
    events = evaluate_events(root, folds, frame_count, association)
    per_frame_events = sum(row["per_frame_detected"] for row in events)
    memory_events = sum(row["memory_detected"] for row in events)
    per_frame_alarms = sum(row["per_frame_alarm_frames"] for row in normal)
    memory_alarms = sum(row["memory_alarm_frames"] for row in normal)
    per_frame_episodes = sum(row["per_frame_alarm_episodes"] for row in normal)
    memory_episodes = sum(row["memory_alarm_episodes"] for row in normal)
    return {
        "experiment": experiment,
        "protocol": {
            "change": "two matching candidate observations in three frames, including scores down to 75% of the frozen threshold",
            "confirmation": "at least one observation crosses the original threshold",
            "reference_map": False,
            "same_route_required": False,
            "memory_scope": "current bag only",
            "association": association,
            "normal_split": fold_protocol,
            "normal_frames": "all frames from five complete organizer bags",
            "event_sequences": "21 exact-mask sequences on held-out bag models",
        },
        "normal": {
            "frames": sum(row["frames"] for row in normal),
            "per_frame_alarm_frames": per_frame_alarms,
            "memory_alarm_frames": memory_alarms,
            "alarm_frame_reduction": 1 - memory_alarms / max(1, per_frame_alarms),
            "per_frame_alarm_episodes": per_frame_episodes,
            "memory_alarm_episodes": memory_episodes,
            "alarm_episode_reduction": 1 - memory_episodes / max(1, per_frame_episodes),
            "by_bag": normal,
        },
        "events": {
            "count": len(events),
            "per_frame_detected": per_frame_events,
            "memory_detected": memory_events,
            "events_lost": per_frame_events - memory_events,
            "rows": events,
        },
        "runtime": {
            "median_ms": statistics.median(latencies),
            "p95_ms": percentile(latencies, 0.95),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--association",
        choices=("world", "range"),
        default="world",
    )
    parser.add_argument("--experiment", default="EXP-073")
    parser.add_argument("--synthetic-frames", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS / "evidence_memory_exp073.json",
    )
    args = parser.parse_args()
    report = run(
        args.dataset_root,
        args.cache,
        args.synthetic_frames,
        args.association,
        args.experiment,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: report[key] for key in ("normal", "events", "runtime")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
