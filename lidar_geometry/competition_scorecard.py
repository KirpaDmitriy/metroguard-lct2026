from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from lidar_geometry.classical_hybrid_experiment import extra_trees
from lidar_geometry.compare_candidate_models import (
    fit_logistic_features,
    linear,
    sample_weights,
    threshold_from_real,
)
from lidar_geometry.detect_obstacles import DetectorConfig, Obstacle
from lidar_geometry.domain_guard import DomainGuard, apply_domain_guard
from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import SafetyDetection, detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.portable_trees import PortableLinearTreeHybrid
from lidar_geometry.risk_model import FEATURE_NAMES, RiskModel, component_features
from lidar_geometry.scenario_catalog import SCENARIOS, matches_target
from lidar_geometry.synthetic import inject_box

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "lidar_geometry" / "artifacts"
DISTANCES_M = (20.0, 40.0, 60.0, 80.0, 100.0)
ALGORITHMS = ("geometry", "g4_linear", "portable_tree_25_75", "ood_guarded_g4")


@dataclass(frozen=True)
class FoldModels:
    linear_predict: object
    linear_model: RiskModel
    tree: ExtraTreesClassifier
    tree_threshold: float
    guard: DomainGuard


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def risk_model_from_predictor(predictor, threshold: float) -> RiskModel:
    parameters = predictor.parameters
    raw_mean = parameters["raw_mean"]
    raw_scale = parameters["raw_scale"]
    return RiskModel(
        FEATURE_NAMES,
        tuple((raw_mean + raw_scale * parameters["mean"]).tolist()),
        tuple((raw_scale * parameters["scale"]).tolist()),
        tuple(parameters["weights"].tolist()),
        float(parameters["bias"]),
        threshold,
    )


def fit_folds(cache_path: Path) -> tuple[dict[str, FoldModels], dict]:
    data = np.load(cache_path, allow_pickle=False)
    features = data["features"].astype(np.float64)
    patches = data["patches"]
    labels = data["labels"]
    groups = data["groups"]
    records = json.loads(str(data["records_json"]))
    folds = {}
    for held_out in sorted(set(groups.tolist())):
        train = groups != held_out
        train_records = [record for record, keep in zip(records, train) if keep]
        weights = sample_weights(labels[train], train_records)
        linear_model = fit_logistic_features(
            features[train], labels[train], weights, linear, steps=1500, l2=0.06
        )
        linear_scores = linear_model.score(features[train])
        linear_threshold = threshold_from_real(linear_scores, train_records)
        tree = extra_trees(100)
        tree.fit(features[train], labels[train], sample_weight=weights)
        hybrid_scores = (
            0.25 * linear_scores + 0.75 * tree.predict_proba(features[train])[:, 1]
        )
        tree_threshold = threshold_from_real(hybrid_scores, train_records)
        real = np.asarray([record["source"] == "real" for record in records])
        guard = DomainGuard.fit(patches[train & real])
        folds[held_out] = FoldModels(
            linear_model.score,
            risk_model_from_predictor(linear_model, linear_threshold),
            tree,
            tree_threshold,
            guard,
        )
    return folds, {
        "split": "leave-one-organizer-recording-out",
        "training_cache": (
            str(cache_path.relative_to(ROOT))
            if cache_path.is_relative_to(ROOT)
            else cache_path.name
        ),
        "threshold": "99th percentile of training real-frame maximum scores",
        "tree": "100 Extra Trees, depth 6, minimum leaf 5, 25/75 linear/tree blend",
        "domain_reference": "training-fold real candidates only",
    }


def decisions(
    cloud,
    geometric: SafetyDetection,
    context: tuple,
    models: FoldModels,
) -> dict[str, tuple[Obstacle, ...]]:
    items = geometric.obstacles
    if items:
        features = np.vstack([component_features(item) for item in items])
        linear_scores = models.linear_predict(features)
        tree_scores = (
            0.25 * linear_scores + 0.75 * models.tree.predict_proba(features)[:, 1]
        )
    else:
        linear_scores = tree_scores = np.empty(0)
    linear_items = tuple(
        item
        for item, score in zip(items, linear_scores)
        if score >= models.linear_model.threshold
    )
    tree_items = tuple(
        item
        for item, score in zip(items, tree_scores)
        if score >= models.tree_threshold
    )
    ranked = filter_geometric_detection(geometric, models.linear_model)
    guarded = apply_domain_guard(
        cloud, ranked, models.linear_model, models.guard, DetectorConfig(), context
    )
    return {
        "geometry": items if geometric.state == "OBSTACLE" else (),
        "g4_linear": linear_items,
        "portable_tree_25_75": tree_items,
        "ood_guarded_g4": guarded.obstacles if guarded.state == "OBSTACLE" else (),
    }


def aggregate_binary(rows: list[dict]) -> dict:
    folds = []
    for bag in ASSUMED_NORMAL:
        selected = [row for row in rows if row["bag"] == bag]
        positive = [row for row in selected if row["expected_alarm"]]
        negative = [row for row in selected if not row["expected_alarm"]]
        folds.append(
            {
                "held_out_bag": bag,
                "positive_cases": len(positive),
                "negative_cases": len(negative),
                **{
                    name: {
                        "positive_recall": sum(row[name] for row in positive)
                        / len(positive),
                        "negative_specificity": sum(not row[name] for row in negative)
                        / len(negative),
                    }
                    for name in ALGORITHMS
                },
            }
        )
    return {
        name: {
            "mean_positive_recall": statistics.mean(
                row[name]["positive_recall"] for row in folds
            ),
            "worst_positive_recall": min(row[name]["positive_recall"] for row in folds),
            "mean_negative_specificity": statistics.mean(
                row[name]["negative_specificity"] for row in folds
            ),
            "folds": [
                {"held_out_bag": row["held_out_bag"], **row[name]} for row in folds
            ],
        }
        for name in ALGORITHMS
    }


def exact_mask_boxes(root: Path, folds: dict[str, FoldModels]) -> dict:
    rows = []
    config = DetectorConfig()
    for bag_index, bag in enumerate(ASSUMED_NORMAL):
        models = folds[bag]
        for frame, cloud in selected_frames(bag_path(root, bag), 50, 2):
            for scenario_index, scenario in enumerate(SCENARIOS):
                for distance in DISTANCES_M:
                    target = replace(scenario.obstacle, distance_m=distance)
                    try:
                        injected = inject_box(
                            cloud,
                            target,
                            config,
                            seed=bag_index * 100_000 + frame * 100 + scenario_index,
                        )
                    except ValueError:
                        continue
                    if injected.visible_points < 3:
                        continue
                    context = []
                    geometric = detect_fast(injected.cloud, config, context_out=context)
                    accepted = decisions(
                        injected.cloud,
                        geometric,
                        context[0] if context else None,
                        models,
                    )
                    rows.append(
                        {
                            "bag": bag,
                            "frame": frame,
                            "scenario": scenario.name,
                            "distance_m": distance,
                            "expected_alarm": scenario.expected_alarm,
                            "visible_points": injected.visible_points,
                            **{
                                name: any(
                                    matches_target(item, target) for item in items
                                )
                                for name, items in accepted.items()
                            },
                        }
                    )
        print(f"scorecard exact boxes: {bag}", flush=True)
    return {
        "protocol": {
            "source_unit": "whole organizer bag",
            "split": "leave-one-organizer-recording-out for learned models",
            "method": "ray-consistent exact-mask AABB insertion",
            "distances_m": list(DISTANCES_M),
            "frames_per_bag": 2,
            "minimum_visible_points": 3,
            "target_attribution": "overlap with the injected AABB",
        },
        "models": aggregate_binary(rows),
        "cases": len(rows),
    }


def count_episodes(states: list[bool]) -> int:
    return sum(
        value and (index == 0 or not states[index - 1])
        for index, value in enumerate(states)
    )


def normal_bags(root: Path, folds: dict[str, FoldModels]) -> dict:
    per_bag = []
    for bag in ASSUMED_NORMAL:
        model = folds[bag]
        states = {name: [] for name in ALGORITHMS}
        for _, cloud in selected_frames(bag_path(root, bag), 1):
            context = []
            geometric = detect_fast(cloud, context_out=context)
            accepted = decisions(
                cloud, geometric, context[0] if context else None, model
            )
            for name, items in accepted.items():
                states[name].append(bool(items))
        per_bag.append(
            {
                "bag": bag,
                "frames": len(next(iter(states.values()))),
                "models": {
                    name: {
                        "alarm_frames": sum(values),
                        "alarm_rate": sum(values) / len(values),
                        "alarm_episodes": count_episodes(values),
                    }
                    for name, values in states.items()
                },
            }
        )
        print(f"scorecard normal bag: {bag}", flush=True)
    total_frames = sum(row["frames"] for row in per_bag)
    return {
        "protocol": {
            "source_unit": "complete organizer bag",
            "sampling": "every frame",
            "labels": "weak-normal label from organizer filenames",
            "split": "each learned model excludes the evaluated bag",
        },
        "models": {
            name: {
                "frames": total_frames,
                "alarm_frames": sum(
                    row["models"][name]["alarm_frames"] for row in per_bag
                ),
                "alarm_rate": sum(
                    row["models"][name]["alarm_frames"] for row in per_bag
                )
                / total_frames,
                "alarm_episodes": sum(
                    row["models"][name]["alarm_episodes"] for row in per_bag
                ),
                "bags": [{"bag": row["bag"], **row["models"][name]} for row in per_bag],
            }
            for name in ALGORITHMS
        },
    }


def overlaps_official_bounds(item: Obstacle, scenario) -> bool:
    target = scenario.obstacle
    return (
        item.lateral_max_m >= target.lateral_m - target.width_m / 2
        and item.lateral_min_m <= target.lateral_m + target.width_m / 2
        and item.height_max_m >= target.bottom_m
        and item.height_min_m <= target.bottom_m + target.height_m
    )


def fixed_models() -> tuple[RiskModel, PortableLinearTreeHybrid, DomainGuard]:
    return (
        RiskModel.load(
            ROOT / "lidar_geometry/models/risk_model_g4_cost_sensitive.json"
        ),
        PortableLinearTreeHybrid.load(ARTIFACTS / "linear_extra_trees_hybrid.npz"),
        DomainGuard.load(ROOT / "lidar_geometry/models/domain_guard.npz"),
    )


def fixed_decisions(
    cloud, geometric, context, models
) -> dict[str, tuple[Obstacle, ...]]:
    linear, tree, guard = models
    items = geometric.obstacles
    if items:
        features = np.vstack([component_features(item) for item in items])
        tree_scores = tree.score(features)
    else:
        tree_scores = np.empty(0)
    ranked = filter_geometric_detection(geometric, linear)
    guarded = apply_domain_guard(
        cloud, ranked, linear, guard, DetectorConfig(), context
    )
    return {
        "geometry": items if geometric.state == "OBSTACLE" else (),
        "g4_linear": ranked.obstacles if ranked.state == "OBSTACLE" else (),
        "portable_tree_25_75": tuple(
            item for item, score in zip(items, tree_scores) if score >= tree.threshold
        ),
        "ood_guarded_g4": guarded.obstacles if guarded.state == "OBSTACLE" else (),
    }


def official_pseudo_labels(bag: Path, motion_path: Path) -> dict:
    motion = load_json(motion_path)["frames_detail"]
    models = fixed_models()
    scenario_rows = [
        {
            "scenario": scenario.name,
            "expected_alarm": scenario.expected_alarm,
            **{
                name: {"visible_frames": 0, "proposal_frames": 0} for name in ALGORITHMS
            },
        }
        for scenario in SCENARIOS[:10]
    ]
    for frame, (_, cloud) in enumerate(iter_bag_messages(bag)):
        context = []
        geometric = detect_fast(cloud, context_out=context)
        accepted = fixed_decisions(
            cloud, geometric, context[0] if context else None, models
        )
        world = motion[frame]["cumulative_m"]
        for index, scenario in enumerate(SCENARIOS[:10]):
            expected_range = 99.25 + index * 100.0 - world
            if not 3.0 <= expected_range <= 120.0:
                continue
            for name, items in accepted.items():
                scenario_rows[index][name]["visible_frames"] += 1
                matched = any(
                    abs(
                        (item.distance_min_m + item.distance_max_m) / 2 - expected_range
                    )
                    <= 2.0
                    and overlaps_official_bounds(item, scenario)
                    for item in items
                )
                scenario_rows[index][name]["proposal_frames"] += matched
        if frame and frame % 250 == 0:
            print(f"scorecard official pseudo-labels: {frame}", flush=True)
    summary = {}
    for name in ALGORITHMS:
        positive = [row for row in scenario_rows if row["expected_alarm"]]
        negative = [row for row in scenario_rows if not row["expected_alarm"]]
        summary[name] = {
            "positive_scenarios_with_proposal": sum(
                row[name]["proposal_frames"] > 0 for row in positive
            ),
            "positive_scenarios": len(positive),
            "positive_coverage": sum(
                row[name]["proposal_frames"] > 0 for row in positive
            )
            / len(positive),
            "negative_scenarios_without_proposal": sum(
                row[name]["proposal_frames"] == 0 for row in negative
            ),
            "negative_scenarios": len(negative),
            "negative_specificity": sum(
                row[name]["proposal_frames"] == 0 for row in negative
            )
            / len(negative),
            "positive_proposal_frames": sum(
                row[name]["proposal_frames"] for row in positive
            ),
        }
    return {
        "protocol": {
            "ground_truth": False,
            "label": "forensic pseudo-label using organizer order and approximate 100 m spacing",
            "warning": "coverage is directional evidence, not a metric claim against ground truth",
            "first_world_m": 99.25,
            "spacing_m": 100.0,
            "range_tolerance_m": 2.0,
        },
        "models": summary,
        "scenarios": scenario_rows,
    }


def composite_shapes() -> dict:
    report = load_json(ARTIFACTS / "shape_generalization.json")
    cache = np.load(ARTIFACTS / "complex_shape_cache_v1.npz", allow_pickle=False)
    records = json.loads(str(cache["records_json"]))
    labels = cache["labels"]
    groups = cache["groups"]
    proposed = np.asarray([record["proposed"] for record in records])
    geometry_recall = statistics.mean(
        proposed[(groups == bag) & (labels == 1)].mean() for bag in sorted(set(groups))
    )
    geometry_specificity = statistics.mean(
        (~proposed[(groups == bag) & (labels == 0)]).mean()
        for bag in sorted(set(groups))
    )
    return {
        "protocol": report["protocol"],
        "models": {
            "geometry": {
                "mean_positive_recall": geometry_recall,
                "mean_negative_specificity": geometry_specificity,
                "source": "proposal decisions in complex_shape_cache_v1.npz",
            },
            "g4_linear": report["models"]["component_linear"],
            "portable_tree_25_75": report["models"]["tree_weighted_75"],
            "ood_guarded_g4": {
                "comparable": False,
                "reason": "the frozen composite cache does not contain context patches for the OOD guard",
            },
        },
    }


def timed_detection(clouds: list, name: str, models) -> list[float]:
    linear, tree, guard = models
    values = []
    for cloud in clouds:
        started = time.perf_counter()
        context = []
        geometric = detect_fast(cloud, context_out=context)
        if name == "g4_linear":
            filter_geometric_detection(geometric, linear)
        elif name == "portable_tree_25_75" and geometric.obstacles:
            tree.score(
                np.vstack([component_features(item) for item in geometric.obstacles])
            )
        elif name == "ood_guarded_g4":
            ranked = filter_geometric_detection(geometric, linear)
            apply_domain_guard(
                cloud,
                ranked,
                linear,
                guard,
                DetectorConfig(),
                context[0] if context else None,
            )
        values.append((time.perf_counter() - started) * 1000)
    return values


def runtime_and_layouts(root: Path) -> dict:
    models = fixed_models()
    timing_bag = bag_path(root, "roundT_doubleT")
    clouds = [cloud for _, cloud in selected_frames(timing_bag, 5, 51)]
    for name in ALGORITHMS:
        timed_detection(clouds[:2], name, models)
    latencies = {name: [] for name in ALGORITHMS}
    for shift in range(len(ALGORITHMS)):
        for index, name in enumerate(ALGORITHMS):
            selected = ALGORITHMS[(index + shift) % len(ALGORITHMS)]
            latencies[selected].extend(timed_detection(clouds, selected, models))
    layout_bags = {
        "xyzi_16": ROOT
        / "external_data/hackathon/synthetic_official/data/cloud_with_fake_obj/cloud_with_fake_obj_0.db3",
        "xyzirt_26": timing_bag,
    }
    layouts = {}
    for layout, path in layout_bags.items():
        _, cloud = next(iter_bag_messages(path))
        outcomes = {}
        for name in ALGORITHMS:
            context = []
            geometric = detect_fast(cloud, context_out=context)
            accepted = fixed_decisions(
                cloud, geometric, context[0] if context else None, models
            )
            outcomes[name] = {
                "supported": True,
                "accepted_components": len(accepted[name]),
            }
        layouts[layout] = {"point_step": cloud.point_step, "models": outcomes}
    return {
        "protocol": {
            "latency": "same 51 XYZIRT frames, bag I/O excluded, four rotated passes after warm-up",
            "hardware_note": "local development host, not organizer container",
            "layout": "one real message per delivered layout through every frozen candidate",
        },
        "models": {
            name: {
                "p50_ms": float(np.quantile(values, 0.50)),
                "p95_ms": float(np.quantile(values, 0.95)),
            }
            for name, values in latencies.items()
        },
        "layouts": layouts,
    }


def accepted_new_variants(registry_path: Path) -> dict:
    registry = {row["id"]: row for row in load_json(registry_path)}
    result = {}
    for experiment_id in ("EXP-065", "EXP-066"):
        row = registry[experiment_id]
        if row["status"] == "complete" and row["decision"] == "accept":
            result[experiment_id] = {
                "title": row["title"],
                "metrics": row["metrics"],
                "artifacts": row["artifacts"],
                "comparison_scope": "experiment-specific frozen paired protocol",
                "incomparable_fields": [
                    "Metrics not emitted by the accepted experiment are not inferred."
                ],
            }
    return result


def choose_default(report: dict) -> tuple[str, list[dict]]:
    ranking = []
    for name in ALGORITHMS:
        normal = report["normal_bags"]["models"][name]
        official = report["official_pseudo_labels"]["models"][name]
        shape = report["composite_unseen_shapes"]["models"][name]
        latency = report["runtime_and_layouts"]["models"][name]
        if normal["alarm_frames"] == 0:
            safety_tier = 0
        elif normal["alarm_rate"] <= 0.001:
            safety_tier = 1
        else:
            safety_tier = 2
        shape_recall = shape.get("mean_positive_recall")
        key = [
            safety_tier,
            normal["alarm_rate"],
            -official["positive_coverage"],
            -(shape_recall if shape_recall is not None else -1.0),
            latency["p95_ms"],
        ]
        ranking.append(
            {
                "algorithm": name,
                "selection_key": key,
                "safety_tier": (
                    "zero"
                    if safety_tier == 0
                    else "near_zero" if safety_tier == 1 else "nonzero"
                ),
                "normal_alarm_rate": normal["alarm_rate"],
                "official_pseudo_positive_coverage": official["positive_coverage"],
                "shape_recall": shape_recall,
                "p95_ms": latency["p95_ms"],
            }
        )
    ranking.sort(key=lambda row: row["selection_key"])
    return ranking[0]["algorithm"], ranking


def evaluate(root: Path) -> dict:
    folds, fold_protocol = fit_folds(ARTIFACTS / "spatial_patch_cache_v1.npz")
    report = {
        "experiment": "EXP-067",
        "selection_policy": {
            "order": [
                "zero/near-zero whole-normal-bag false alarms",
                "official pseudo-label positive scenario coverage",
                "unseen composite-shape recall",
                "p95 latency",
            ],
            "official_labels_are_ground_truth": False,
            "missing_values": "incomparable; never imputed",
            "runtime_changed_before_selection": False,
        },
        "fold_training": fold_protocol,
        "exact_mask_boxes": exact_mask_boxes(root, folds),
        "normal_bags": normal_bags(root, folds),
        "composite_unseen_shapes": composite_shapes(),
        "official_pseudo_labels": official_pseudo_labels(
            ROOT
            / "external_data/hackathon/synthetic_official/data/cloud_with_fake_obj/cloud_with_fake_obj_0.db3",
            ARTIFACTS / "official_ego_motion.json",
        ),
        "runtime_and_layouts": runtime_and_layouts(root),
        "accepted_new_variants": accepted_new_variants(
            ROOT / "experiments/registry.json"
        ),
    }
    selected, ranking = choose_default(report)
    report["selection"] = {
        "recommended_default": selected,
        "ranking": ranking,
        "alternatives": [row["algorithm"] for row in ranking[1:]],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=ROOT / "external_data/hackathon/for_hackathon",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS / "competition_scorecard_exp067.json",
    )
    args = parser.parse_args()
    report = evaluate(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["selection"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
