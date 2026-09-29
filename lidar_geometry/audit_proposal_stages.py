from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics

from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box
from lidar_geometry.temporal import WorldComponentTracker


def has_target(detection, target) -> bool:
    return any(matches_target(component, target) for component in detection.obstacles)


def target_only(detection, target):
    matched = tuple(
        component for component in detection.obstacles if matches_target(component, target)
    )
    return replace(
        detection,
        state="OBSTACLE" if matched else "UNKNOWN",
        obstacle=bool(matched),
        nearest_distance_m=min((item.distance_min_m for item in matched), default=None),
        obstacles=matched,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=Path(
        "lidar_geometry/artifacts/proposal_stage_audit.json"
    ))
    args = parser.parse_args()
    model = replace(RiskModel.load(args.model), threshold=args.threshold)

    rows = []
    positives = [scenario for scenario in SCENARIOS if scenario.expected_alarm]
    for bag_index, name in enumerate(ASSUMED_NORMAL[:3]):
        clouds = list(selected_frames(bag_path(args.dataset_root, name), 1, args.frames))
        for scenario_index, scenario in enumerate(positives):
            tracker = WorldComponentTracker()
            observable = proposal = ranked = tracked = 0
            target_scores = []
            for step, (_frame, cloud) in enumerate(clouds):
                target = replace(scenario.obstacle, distance_m=40.0 - step)
                injected = inject_box(
                    cloud,
                    target,
                    seed=bag_index * 10_000 + scenario_index * 100 + step,
                )
                observable += injected.visible_points >= 3
                geometric = detect_fast(injected.cloud)
                proposal += has_target(geometric, target)
                target_scores.extend(
                    model.score(component)
                    for component in geometric.obstacles
                    if matches_target(component, target)
                )
                filtered = filter_geometric_detection(geometric, model)
                ranked += has_target(filtered, target)
                tracked += tracker.update(target_only(filtered, target), float(step)).obstacle
            rows.append({
                "bag": name,
                "scenario": scenario.name,
                "frames": len(clouds),
                "observable_frames": observable,
                "proposal_frames": proposal,
                "ranked_frames": ranked,
                "tracked_frames": tracked,
                "target_score_median": (
                    statistics.median(target_scores) if target_scores else None
                ),
                "target_score_max": max(target_scores, default=None),
            })

    totals = {
        key: sum(row[key] for row in rows)
        for key in ("frames", "observable_frames", "proposal_frames", "ranked_frames", "tracked_frames")
    }
    by_scenario = {}
    for scenario in positives:
        selected = [row for row in rows if row["scenario"] == scenario.name]
        by_scenario[scenario.name] = {
            key: sum(row[key] for row in selected)
            for key in ("frames", "observable_frames", "proposal_frames", "ranked_frames", "tracked_frames")
        }
        scores = [
            row["target_score_median"]
            for row in selected
            if row["target_score_median"] is not None
        ]
        by_scenario[scenario.name]["sequence_median_score"] = (
            statistics.median(scores) if scores else None
        )
        by_scenario[scenario.name]["max_score"] = max(
            (row["target_score_max"] for row in selected if row["target_score_max"] is not None),
            default=None,
        )
    report = {
        "protocol": {
            "target_attribution": "strict overlap with known inserted bounds",
            "threshold": args.threshold,
            "motion": "one metre per frame matching synthetic approach",
        },
        "totals": totals,
        "by_scenario": by_scenario,
        "sequences": rows,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"totals": totals, "by_scenario": by_scenario}, indent=2))


if __name__ == "__main__":
    main()
