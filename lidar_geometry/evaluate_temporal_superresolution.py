from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from lidar_geometry.evaluate_mvp import ASSUMED_NORMAL, bag_path, selected_frames
from lidar_geometry.fast_detector import cloud_arrays
from lidar_geometry.scenario_catalog import SCENARIOS
from lidar_geometry.synthetic import inject_box


def target_world_voxels(injected, motion_m: float, voxel_m: float) -> set[tuple[int, int, int]]:
    if not injected.modified_indices:
        return set()
    arrays = cloud_arrays(injected.cloud)
    indexes = np.asarray(injected.modified_indices)
    world = motion_m - arrays["y"][indexes]
    lateral = arrays["x"][indexes] - injected.track_center_m
    height = arrays["z"][indexes] - injected.rail_z_m
    cells = np.rint(np.column_stack((world, lateral, height)) / voxel_m).astype(np.int32)
    return {tuple(int(value) for value in row) for row in cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--voxel-m", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path(
        "lidar_geometry/artifacts/temporal_superresolution_feasibility.json"
    ))
    args = parser.parse_args()

    rows = []
    positives = [scenario for scenario in SCENARIOS if scenario.expected_alarm]
    for bag_index, name in enumerate(ASSUMED_NORMAL[:3]):
        clouds = list(selected_frames(bag_path(args.dataset_root, name), 1, args.frames))
        for scenario_index, scenario in enumerate(positives):
            accumulated = set()
            visible_per_frame = []
            voxels_per_frame = []
            for step, (_frame, cloud) in enumerate(clouds):
                target = replace(scenario.obstacle, distance_m=40.0 - step)
                injected = inject_box(
                    cloud,
                    target,
                    seed=bag_index * 10_000 + scenario_index * 100 + step,
                )
                voxels = target_world_voxels(injected, float(step), args.voxel_m)
                accumulated.update(voxels)
                visible_per_frame.append(injected.visible_points)
                voxels_per_frame.append(len(voxels))
            rows.append({
                "bag": name,
                "scenario": scenario.name,
                "max_visible_points_in_frame": max(visible_per_frame, default=0),
                "max_target_voxels_in_frame": max(voxels_per_frame, default=0),
                "accumulated_target_voxels": len(accumulated),
                "frames_with_any_return": sum(value > 0 for value in visible_per_frame),
            })

    report = {
        "protocol": {
            "purpose": "oracle-mask feasibility only, not detector quality",
            "motion": "one metre per frame matching the synthetic approach",
            "voxel_m": args.voxel_m,
            "target_mask_runtime_feature": False,
        },
        "summary": {
            "sequences": len(rows),
            "never_observed": sum(row["accumulated_target_voxels"] == 0 for row in rows),
            "single_frame_below_3_but_accumulated_at_least_3": sum(
                row["max_visible_points_in_frame"] < 3
                and row["accumulated_target_voxels"] >= 3
                for row in rows
            ),
            "median_accumulation_gain": float(np.median([
                row["accumulated_target_voxels"]
                / max(1, row["max_target_voxels_in_frame"])
                for row in rows
            ])),
        },
        "sequences": rows,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
