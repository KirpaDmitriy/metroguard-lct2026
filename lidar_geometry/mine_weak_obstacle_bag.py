from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lidar_geometry.domain_guard import component_patch
from lidar_geometry.ego_motion import EgoMotionEstimator
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.invariant_spatial_experiment import random_convolution_descriptor
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.risk_model import RiskModel, component_features


def nearest_distance(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale[scale < 1e-4] = 1.0
    reference = np.clip((reference - mean) / scale, -8, 8)
    query = np.clip((query - mean) / scale, -8, 8)
    result = np.empty(len(query))
    for start in range(0, len(query), 128):
        block = query[start:start + 128]
        distance = ((block[:, None] - reference[None]) ** 2).mean(axis=2)
        result[start:start + len(block)] = distance.min(axis=1)
    return result


def render(rows: list[dict], patches: np.ndarray, output: Path) -> None:
    width = 3 * 96 + 220
    height = max(1, len(rows)) * 104
    canvas = Image.new("RGB", (width, height), "#0b1020")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        patch = patches[row["representative_index"]]
        for channel in range(3):
            image = Image.fromarray(np.uint8(np.clip(patch[channel], 0, 1) * 255))
            image = image.resize((96, 96), Image.Resampling.NEAREST).convert("RGB")
            canvas.paste(image, (channel * 96, row_index * 104))
        text = (
            f"#{row_index + 1} frames={row['frames']}\n"
            f"world={row['world_m']:.1f}m lat={row['lateral_m']:.2f}\n"
            f"novelty={row['novelty']:.2f} risk={row['risk']:.2f}"
        )
        draw.multiline_text((296, row_index * 104 + 8), text, fill="#e5e7eb", spacing=5)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def mine(
    bag: Path,
    normal_cache: Path,
    model_path: Path,
    output: Path,
    image: Path,
    cache_output: Path,
) -> None:
    cached = np.load(normal_cache, allow_pickle=False)
    groups = cached["groups"]
    reference_mask = np.asarray(["doubleT" in value for value in groups])
    reference = random_convolution_descriptor(cached["patches"][reference_mask])
    model = RiskModel.load(model_path)
    motion = EgoMotionEstimator()
    patches = []
    records = []
    for frame, (_, cloud) in enumerate(iter_bag_messages(bag)):
        pose = motion.update(cloud)
        context = []
        detection = detect_fast(cloud, context_out=context)
        if not context:
            continue
        for item in detection.obstacles:
            patches.append(component_patch(context[0], item))
            records.append({
                "frame": frame,
                "world_m": pose.cumulative_m + (item.distance_min_m + item.distance_max_m) / 2,
                "lateral_m": (item.lateral_min_m + item.lateral_max_m) / 2,
                "risk": model.score(item),
                "component": component_features(item).tolist(),
            })
    patch_array = np.asarray(patches, dtype=np.float32)
    descriptor = random_convolution_descriptor(patch_array)
    novelty = nearest_distance(reference, descriptor)
    tracks: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(records):
        key = round(record["world_m"] / 1.5), round(record["lateral_m"] / 0.3)
        tracks.setdefault(key, []).append(index)
    rows = []
    for indexes in tracks.values():
        frames = len(set(records[index]["frame"] for index in indexes))
        if frames < 2:
            continue
        representative = max(indexes, key=lambda index: novelty[index])
        rows.append({
            "frames": frames,
            "world_m": float(np.median([records[index]["world_m"] for index in indexes])),
            "lateral_m": float(np.median([records[index]["lateral_m"] for index in indexes])),
            "novelty": float(np.median(novelty[indexes])),
            "risk": float(np.median([records[index]["risk"] for index in indexes])),
            "representative_frame": records[representative]["frame"],
            "representative_index": representative,
            "indexes": indexes,
        })
    rows.sort(key=lambda row: row["novelty"] * np.log1p(row["frames"]), reverse=True)
    hotspots = []
    for row in rows:
        if row["novelty"] < 0.35:
            continue
        if all(abs(row["world_m"] - value) >= 5.0 for value in hotspots):
            hotspots.append(row["world_m"])
        if len(hotspots) == 2:
            break
    selected_tracks = [
        row for row in rows
        if row["frames"] >= 5
        and row["novelty"] >= 0.35
        and any(abs(row["world_m"] - value) <= 2.0 for value in hotspots)
    ]
    selected_indexes = sorted({
        index for row in selected_tracks for index in row["indexes"]
    })
    public_rows = [
        {key: value for key, value in row.items() if key != "indexes"}
        for row in rows
    ]
    report = {
        "protocol": {
            "label": "positive bag only; instance labels unknown",
            "normal_memory": "doubleT organizer patches excluding obstacle bag",
            "track_grid_m": [1.5, 0.3],
            "selection": "persistence times nearest-memory novelty",
        },
        "frames": 201,
        "candidates": len(records),
        "persistent_tracks": len(rows),
        "hotspots_m": hotspots,
        "selected_tracks": len(selected_tracks),
        "selected_candidates": len(selected_indexes),
        "top_tracks": public_rows[:20],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    render(public_rows[:12], patch_array, image)
    cache_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_output,
        patches=patch_array[selected_indexes],
        records_json=json.dumps([records[index] for index in selected_indexes]),
        hotspots=np.asarray(hotspots),
    )
    print(json.dumps({key: value for key, value in report.items() if key != "top_tracks"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument(
        "--normal-cache",
        type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--cache-output",
        type=Path,
        default=Path("lidar_geometry/artifacts/weak_obstacle_candidates.npz"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("lidar_geometry/models/risk_model.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/weak_obstacle_mining.json"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("lidar_geometry/artifacts/weak_obstacle_mining.png"),
    )
    args = parser.parse_args()
    mine(
        args.bag,
        args.normal_cache,
        args.model,
        args.output,
        args.image,
        args.cache_output,
    )


if __name__ == "__main__":
    main()
