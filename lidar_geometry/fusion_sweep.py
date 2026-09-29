from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from lidar_geometry.detect_obstacles import Obstacle
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import CORE_HALF_WIDTH_M, detect_fast
from lidar_geometry.risk_model import RiskModel


def obstacle_from_dict(payload: dict) -> Obstacle:
    return Obstacle(**payload)


def obvious_intrusion(item: Obstacle, minimum_span_m: float | None) -> bool:
    if minimum_span_m is None:
        return False
    lateral_center = (item.lateral_min_m + item.lateral_max_m) / 2
    lateral_span = item.lateral_max_m - item.lateral_min_m
    height_span = item.height_max_m - item.height_min_m
    return (
        abs(lateral_center) <= CORE_HALF_WIDTH_M
        and max(lateral_span, height_span) >= minimum_span_m
    )


def alarm(item: Obstacle, score: float, threshold: float, span: float | None) -> bool:
    return score >= threshold or obvious_intrusion(item, span)


def collect_real_frames(root: Path, every: int, model: RiskModel) -> list[dict]:
    rows = []
    for bag in ASSUMED_NORMAL:
        for frame, cloud in selected_frames(bag_path(root, bag), every):
            result = detect_fast(cloud)
            rows.append({
                "bag": bag,
                "frame": frame,
                "components": [
                    {"component": item, "score": model.score(item)}
                    for item in result.obstacles
                ],
            })
    return rows


def evaluate_operating_point(
    scenarios: list[dict],
    real_frames: list[dict],
    threshold: float,
    span: float | None,
) -> dict:
    scenario_counts = defaultdict(lambda: [0, 0])
    for row in scenarios:
        if not row["evaluable"]:
            continue
        if row.get("matched_component") is None:
            predicted = False
        else:
            item = obstacle_from_dict(row["matched_component"])
            predicted = alarm(item, float(row["target_score"]), threshold, span)
        correct = predicted == row["expected_alarm"]
        key = "positive" if row["expected_alarm"] else "negative"
        scenario_counts[(key, row["distance_m"])][0] += int(correct)
        scenario_counts[(key, row["distance_m"])][1] += 1

    bag_counts = defaultdict(lambda: [0, 0])
    for row in real_frames:
        predicted = any(
            alarm(value["component"], value["score"], threshold, span)
            for value in row["components"]
        )
        bag_counts[row["bag"]][0] += int(predicted)
        bag_counts[row["bag"]][1] += 1

    positive = [value for (kind, _), value in scenario_counts.items() if kind == "positive"]
    negative = [value for (kind, _), value in scenario_counts.items() if kind == "negative"]
    real_alarm_frames = sum(value[0] for value in bag_counts.values())
    real_frames_total = sum(value[1] for value in bag_counts.values())
    return {
        "threshold": threshold,
        "obvious_intrusion_span_m": span,
        "scenario_positive_recall": sum(v[0] for v in positive) / sum(v[1] for v in positive),
        "scenario_negative_specificity": sum(v[0] for v in negative) / sum(v[1] for v in negative),
        "assumed_normal_frame_alarm_rate": real_alarm_frames / real_frames_total,
        "by_distance": {
            f"{kind}_{distance:g}m": correct / total
            for (kind, distance), (correct, total) in sorted(scenario_counts.items())
        },
        "by_bag_alarm_rate": {
            bag: alarms / frames for bag, (alarms, frames) in sorted(bag_counts.items())
        },
    }


def is_dominated(candidate: dict, others: list[dict]) -> bool:
    for other in others:
        no_worse = (
            other["scenario_positive_recall"] >= candidate["scenario_positive_recall"]
            and other["scenario_negative_specificity"] >= candidate["scenario_negative_specificity"]
            and other["assumed_normal_frame_alarm_rate"] <= candidate["assumed_normal_frame_alarm_rate"]
        )
        strictly_better = (
            other["scenario_positive_recall"] > candidate["scenario_positive_recall"]
            or other["scenario_negative_specificity"] > candidate["scenario_negative_specificity"]
            or other["assumed_normal_frame_alarm_rate"] < candidate["assumed_normal_frame_alarm_rate"]
        )
        if no_worse and strictly_better:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("scenario_report", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/fusion_sweep.json"),
    )
    args = parser.parse_args()
    model = RiskModel.load(args.model)
    scenarios = json.loads(args.scenario_report.read_text(encoding="utf-8"))["records"]
    for row in scenarios:
        component = row.get("matched_component")
        row["target_score"] = (
            model.score(obstacle_from_dict(component)) if component is not None else None
        )
    real_frames = collect_real_frames(args.dataset_root, args.every, model)
    thresholds = sorted({0.5, 0.6, 0.7, 0.8, 0.9, round(model.threshold, 6)})
    spans = (None, 0.8, 1.0, 1.2)
    operating_points = [
        evaluate_operating_point(scenarios, real_frames, threshold, span)
        for threshold in thresholds for span in spans
    ]
    frontier = [point for point in operating_points if not is_dominated(point, operating_points)]
    frontier.sort(key=lambda item: (
        item["assumed_normal_frame_alarm_rate"],
        -item["scenario_positive_recall"],
    ))
    report = {
        "protocol": {
            "scenario_source": str(args.scenario_report),
            "model": str(args.model),
            "normal_label_is_weak": True,
            "normal_split_unit": "whole organizer bag",
            "sample_every": args.every,
            "obvious_intrusion": "central component with lateral or vertical span above threshold",
        },
        "pareto_frontier": frontier,
        "operating_points": operating_points,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pareto_frontier": frontier}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
