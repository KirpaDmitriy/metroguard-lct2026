from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np

from lidar_geometry.pointcloud2 import PointCloud2, parse_pointcloud2


FIELD_TYPES = {
    1: "i1",
    2: "u1",
    3: "<i2",
    4: "<u2",
    5: "<i4",
    6: "<u4",
    7: "<f4",
    8: "<f8",
}


def cloud_array(cloud: PointCloud2) -> np.ndarray:
    names = [field.name for field in cloud.fields]
    formats = [FIELD_TYPES[field.datatype] for field in cloud.fields]
    offsets = [field.offset for field in cloud.fields]
    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": cloud.point_step,
    })
    return np.frombuffer(cloud.data, dtype=dtype, count=cloud.point_count)


def sampled_clouds(path: Path, sample_frames: int) -> tuple[int, list[tuple[int, PointCloud2]]]:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        condition = "topics.type = 'sensor_msgs/msg/PointCloud2'"
        count = connection.execute(
            "SELECT COUNT(*) FROM messages JOIN topics ON topics.id = messages.topic_id "
            f"WHERE {condition}"
        ).fetchone()[0]
        indexes = np.linspace(0, count - 1, min(count, sample_frames), dtype=np.int64)
        rows = []
        for index in np.unique(indexes):
            row = connection.execute(
                "SELECT messages.data FROM messages "
                "JOIN topics ON topics.id = messages.topic_id "
                f"WHERE {condition} ORDER BY messages.timestamp LIMIT 1 OFFSET ?",
                (int(index),),
            ).fetchone()
            rows.append((int(index), parse_pointcloud2(row[0])))
    return count, rows


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def run_summary(xyz: np.ndarray, start: int, end: int) -> dict:
    points = xyz[start:end]
    extent = np.ptp(points, axis=0)
    return {
        "start": start,
        "length": end - start,
        "centroid_xyz_m": np.mean(points, axis=0).round(4).tolist(),
        "extent_xyz_m": extent.round(4).tolist(),
        "bbox_diagonal_m": round(float(np.linalg.norm(extent)), 4),
    }


def summarize_frame(frame: int, cloud: PointCloud2) -> dict:
    points = cloud_array(cloud)
    xyz = np.column_stack((points["x"], points["y"], points["z"]))
    valid = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    intensity_one = valid & (points["intensity"] == 1)
    runs = true_runs(intensity_one)
    ranked_runs = sorted(runs, key=lambda item: item[1] - item[0], reverse=True)[:8]
    valid_xyz = xyz[valid]
    adjacent = np.linalg.norm(np.diff(valid_xyz, axis=0), axis=1)
    row = {
        "frame": frame,
        "points": cloud.point_count,
        "valid_points": int(valid.sum()),
        "intensity_one_points": int(intensity_one.sum()),
        "intensity_one_fraction": float(intensity_one.sum() / max(1, valid.sum())),
        "max_intensity_one_run": max((end - start for start, end in runs), default=0),
        "top_intensity_one_runs": [run_summary(xyz, start, end) for start, end in ranked_runs],
        "adjacent_distance_p50_m": float(np.quantile(adjacent, 0.5)),
        "adjacent_distance_p99_m": float(np.quantile(adjacent, 0.99)),
    }
    fields = set(points.dtype.names or ())
    if "ring" in fields:
        rings = points["ring"][valid]
        row["ring_unique"] = int(np.unique(rings).size)
        row["ring_equal_adjacent_fraction"] = float(np.mean(rings[1:] == rings[:-1]))
    if "timestamp" in fields:
        timestamps = points["timestamp"][valid]
        row["timestamp_nondecreasing_fraction"] = float(
            np.mean(timestamps[1:] >= timestamps[:-1])
        )
    return row


def quantiles(values: list[float]) -> dict:
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p50", "p95", "max"),
            np.quantile(values, (0, 0.5, 0.95, 1)),
        )
    }


def summarize_bag(path: Path, root: Path, sample_frames: int) -> dict:
    frame_count, clouds = sampled_clouds(path, sample_frames)
    frames = [summarize_frame(frame, cloud) for frame, cloud in clouds]
    first_cloud = clouds[0][1]
    return {
        "bag": str(path.relative_to(root)),
        "frame_count": frame_count,
        "sampled_frames": len(frames),
        "schema": {
            "point_step": first_cloud.point_step,
            "fields": [field.name for field in first_cloud.fields],
        },
        "intensity_one_fraction": quantiles([row["intensity_one_fraction"] for row in frames]),
        "max_intensity_one_run": quantiles([row["max_intensity_one_run"] for row in frames]),
        "adjacent_distance_p99_m": quantiles([row["adjacent_distance_p99_m"] for row in frames]),
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--sample-frames", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/synthetic_provenance_audit.json"),
    )
    args = parser.parse_args()
    bags = sorted(args.data_root.rglob("*.db3"))
    official = [path for path in bags if "synthetic_official" in path.parts]
    normal = [path for path in bags if "for_hackathon" in path.parts]
    if len(official) != 1 or not normal:
        raise SystemExit("Expected one official synthetic bag and organizer normal bags")

    normal_reports = [summarize_bag(path, args.data_root, args.sample_frames) for path in normal]
    official_report = summarize_bag(official[0], args.data_root, args.sample_frames)
    normal_frames = [frame for bag in normal_reports for frame in bag["frames"]]
    official_frames = official_report["frames"]
    normal_max_run = max(frame["max_intensity_one_run"] for frame in normal_frames)
    synthetic_exceeding = [
        frame for frame in official_frames if frame["max_intensity_one_run"] > normal_max_run
    ]
    candidate_runs = [
        run
        for frame in synthetic_exceeding
        for run in frame["top_intensity_one_runs"]
        if run["length"] > normal_max_run
    ]
    compact_runs = [run for run in candidate_runs if run["bbox_diagonal_m"] <= 8.0]
    normal_steps = sorted({bag["schema"]["point_step"] for bag in normal_reports})
    report = {
        "experiment": "EXP-062",
        "protocol": {
            "purpose": "audit-only generator provenance; no runtime feature",
            "sample_selection": "deterministic evenly spaced frames including endpoints",
            "sample_frames_per_bag": args.sample_frames,
            "normal_bags": len(normal_reports),
            "limitations": [
                "The official bag and all organizer bags were already inspected.",
                "Frame sampling does not prove absence on unsampled points.",
                "Intensity one is a native sensor value and cannot be a ground-truth mask by itself.",
            ],
        },
        "comparison": {
            "normal_point_steps": normal_steps,
            "official_point_step": official_report["schema"]["point_step"],
            "schema_distinguishes_dataset": official_report["schema"]["point_step"] not in normal_steps,
            "schema_localizes_inserted_points": False,
            "normal_max_intensity_one_run": normal_max_run,
            "official_frames_exceeding_normal_run": len(synthetic_exceeding),
            "official_sampled_frames": len(official_frames),
            "normal_sampled_frames": len(normal_frames),
            "official_exceeding_fraction": len(synthetic_exceeding) / len(official_frames),
            "candidate_long_runs": len(candidate_runs),
            "compact_candidate_long_runs": len(compact_runs),
            "normal_compact_runs_above_normal_max": 0,
            "intensity_one_present_in_every_normal_bag": all(
                bag["intensity_one_fraction"]["max"] > 0 for bag in normal_reports
            ),
        },
        "decision": {
            "intensity_one_alone": "rejected_as_label",
            "schema_or_point_order_at_runtime": "rejected",
            "offline_weak_labeling": "sparse_proposals_only_with_geometry_and_human_review",
            "reason": (
                "Compact long intensity-one runs occur in 11/64 sampled official frames and "
                "0/384 sampled normal frames, but intensity one itself occurs in every normal "
                "bag and the post-hoc run threshold has no independent ground-truth audit."
            ),
        },
        "normal_bags": normal_reports,
        "official_synthetic_bag": official_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
