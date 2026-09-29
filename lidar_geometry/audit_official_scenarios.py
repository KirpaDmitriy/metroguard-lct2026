from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.fast_detector import SafetyDetection
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.scenario_catalog import SCENARIOS
from lidar_geometry.temporal import ComponentTracker, WorldComponentTracker


def obstacle(payload: dict) -> Obstacle:
    return Obstacle(**payload)


def detection(components: tuple[Obstacle, ...], confidence: float) -> SafetyDetection:
    return SafetyDetection(
        state="OBSTACLE" if components else "UNKNOWN",
        obstacle=bool(components),
        nearest_distance_m=min((item.distance_min_m for item in components), default=None),
        confidence=confidence,
        observability=1.0,
        candidate_points=sum(item.points for item in components),
        track_bins=0,
        observed_track_bins=0,
        reliable_range_min_m=None,
        reliable_range_max_m=None,
        obstacles=components,
        reason="official_world_coordinate_attribution",
    )


def overlaps_scenario_bounds(item: Obstacle, scenario) -> bool:
    target = scenario.obstacle
    lateral_min = target.lateral_m - target.width_m / 2
    lateral_max = target.lateral_m + target.width_m / 2
    height_min = target.bottom_m
    height_max = target.bottom_m + target.height_m
    return (
        item.lateral_max_m >= lateral_min
        and item.lateral_min_m <= lateral_max
        and item.height_max_m >= height_min
        and item.height_min_m <= height_max
    )


def evaluate(
    frames: list[dict],
    motion: list[dict],
    model: RiskModel,
    threshold: float,
    first_world_m: float,
    spacing_m: float,
    tolerance_m: float,
    world_tracking: bool = False,
) -> dict:
    scenario_rows = []
    for index, scenario in enumerate(SCENARIOS[:10]):
        world_m = first_world_m + index * spacing_m
        tracker = (
            WorldComponentTracker(required_hits=2)
            if world_tracking else ComponentTracker(required_hits=2)
        )
        visible_frames = 0
        proposal_frames = 0
        accepted_frames = 0
        tracked_frames = 0
        first_detection_range = None
        maximum_score = 0.0
        attributed_points = 0
        for row, pose in zip(frames, motion):
            expected_range = world_m - pose["cumulative_m"]
            if not 3.0 <= expected_range <= 120.0:
                continue
            visible_frames += 1
            nearby = []
            accepted = []
            for payload in row["components"]:
                item = obstacle(payload)
                center = (item.distance_min_m + item.distance_max_m) / 2
                if abs(center - expected_range) > tolerance_m:
                    continue
                if not overlaps_scenario_bounds(item, scenario):
                    continue
                nearby.append(item)
                score = model.score(item)
                maximum_score = max(maximum_score, score)
                if score >= threshold:
                    accepted.append(item)
                    attributed_points += item.points
            proposal_frames += bool(nearby)
            accepted_frames += bool(accepted)
            proposal = detection(tuple(accepted), maximum_score)
            decision = (
                tracker.update(proposal, pose["cumulative_m"])
                if world_tracking else tracker.update(proposal)
            )
            if decision.obstacle:
                tracked_frames += 1
                if first_detection_range is None:
                    first_detection_range = expected_range
        predicted_alarm = tracked_frames > 0
        scenario_rows.append({
            "scenario": scenario.name,
            "expected_alarm": scenario.expected_alarm,
            "world_m": world_m,
            "visible_frames": visible_frames,
            "proposal_frames": proposal_frames,
            "accepted_frames": accepted_frames,
            "tracked_frames": tracked_frames,
            "predicted_alarm": predicted_alarm,
            "correct": predicted_alarm == scenario.expected_alarm,
            "first_detection_range_m": first_detection_range,
            "maximum_score": maximum_score,
            "attributed_points": attributed_points,
        })
    positives = [row for row in scenario_rows if row["expected_alarm"]]
    negatives = [row for row in scenario_rows if not row["expected_alarm"]]
    return {
        "threshold": threshold,
        "scenario_recall": sum(row["predicted_alarm"] for row in positives) / len(positives),
        "negative_specificity": sum(not row["predicted_alarm"] for row in negatives) / len(negatives),
        "correct_scenarios": sum(row["correct"] for row in scenario_rows),
        "scenarios": scenario_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_report", type=Path)
    parser.add_argument("motion_report", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--first-world", type=float, default=99.25)
    parser.add_argument("--spacing", type=float, default=100.0)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--world-tracker", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/official_scenario_audit.json"),
    )
    args = parser.parse_args()
    official = json.loads(args.official_report.read_text(encoding="utf-8"))
    motion = json.loads(args.motion_report.read_text(encoding="utf-8"))["frames_detail"]
    model = RiskModel.load(args.model)
    thresholds = sorted({0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, model.threshold})
    operating_points = [
        evaluate(
            official["frames_detail"], motion, replace(model, threshold=threshold),
            threshold, args.first_world, args.spacing, args.tolerance,
            args.world_tracker,
        )
        for threshold in thresholds
    ]
    report = {
        "protocol": {
            "blind": False,
            "status": "range-attributed audit with organizer-provided order and spacing",
            "first_world_m": args.first_world,
            "spacing_m": args.spacing,
            "range_tolerance_m": args.tolerance,
            "motion_source": str(args.motion_report),
            "model": str(args.model),
            "tracker": "world_coordinate" if args.world_tracker else "range_lateral",
        },
        "operating_points": operating_points,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "operating_points": [{
            key: point[key] for key in (
                "threshold", "scenario_recall", "negative_specificity", "correct_scenarios"
            )
        } for point in operating_points],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
