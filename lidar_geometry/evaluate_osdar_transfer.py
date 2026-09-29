from __future__ import annotations

import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.pointcloud2 import PointCloud2, PointField
from lidar_geometry.risk_model import RiskModel


FIELDS = (
    PointField("x", 0, 7, 1),
    PointField("y", 4, 7, 1),
    PointField("z", 8, 7, 1),
    PointField("intensity", 12, 7, 1),
)


def read_ascii_pcd(stream) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    fields = None
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PCD header has no DATA line")
        decoded = line.decode("ascii").strip()
        if decoded.startswith("FIELDS "):
            fields = tuple(decoded.split()[1:])
        if decoded == "DATA ascii":
            break
        if decoded.startswith("DATA "):
            raise ValueError(f"Unsupported PCD encoding: {decoded}")
    if fields is None:
        raise ValueError("PCD header has no FIELDS line")
    values = np.loadtxt(stream, dtype=np.float32, ndmin=2)
    return {name: values[:, index] for index, name in enumerate(fields)}, fields


def metro_cloud(values: dict[str, np.ndarray], *, pandar_only: bool) -> PointCloud2:
    keep = np.ones(len(values["x"]), dtype=bool)
    if pandar_only and "sensor_index" in values:
        keep &= values["sensor_index"] == 0
    output = np.empty(np.count_nonzero(keep), dtype={
        "names": ("x", "y", "z", "intensity"),
        "formats": ("<f4", "<f4", "<f4", "<f4"),
        "offsets": (0, 4, 8, 12),
        "itemsize": 16,
    })
    output["x"] = values["y"][keep]
    output["y"] = -values["x"][keep]
    output["z"] = values["z"][keep]
    output["intensity"] = values.get("intensity", np.zeros(len(keep)))[keep]
    data = memoryview(output).cast("B")
    return PointCloud2(
        0, 0, "osdar_lidar", 1, len(output), FIELDS, False, 16,
        len(data), data, False,
    )


def target_bounds(values: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    if "label" not in values:
        return None
    target = values["label"] == 1
    if np.count_nonzero(target) < 3:
        return None
    points = np.column_stack((
        values["x"][target],
        values["y"][target],
    ))
    return points.min(axis=0), points.max(axis=0)


def overlaps_target(item, bounds, tolerance: float = 0.05) -> bool:
    minimum, maximum = bounds
    return (
        item.distance_max_m >= minimum[0] - tolerance
        and item.distance_min_m <= maximum[0] + tolerance
        and item.lateral_max_m >= minimum[1] - tolerance
        and item.lateral_min_m <= maximum[1] + tolerance
    )


def evaluate(values, model: RiskModel, pandar_only: bool) -> dict:
    cloud = metro_cloud(values, pandar_only=pandar_only)
    geometric = detect_fast(cloud)
    ranked = filter_geometric_detection(geometric, model)
    bounds = target_bounds(values)
    target_in_nominal_clearance = bool(
        bounds is not None
        and bounds[1][0] >= 3.0
        and bounds[0][0] <= 150.0
        and bounds[1][1] >= -1.05
        and bounds[0][1] <= 1.05
    )
    return {
        "points": cloud.point_count,
        "geometric_state": geometric.state,
        "ranked_state": ranked.state,
        "observability": geometric.observability,
        "track_bins": geometric.track_bins,
        "target_points": int(np.count_nonzero(values.get("label", np.zeros(0)) == 1)),
        "target_bounds_forward_lateral": (
            [bounds[0].tolist(), bounds[1].tolist()] if bounds is not None else None
        ),
        "target_in_nominal_clearance": target_in_nominal_clearance,
        "geometric_target": (
            any(overlaps_target(item, bounds) for item in geometric.obstacles)
            if bounds is not None else None
        ),
        "ranked_target": (
            any(overlaps_target(item, bounds) for item in ranked.obstacles)
            if bounds is not None else None
        ),
    }


def real_rows(root: Path, model: RiskModel) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/lidar/*.pcd")):
        with path.open("rb") as stream:
            values, _ = read_ascii_pcd(stream)
        rows.append({"source": str(path), **evaluate(values, model, True)})
    return rows


def augmented_rows(archives: list[Path], model: RiskModel, samples: int) -> list[dict]:
    rows = []
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            names = [name for name in source.namelist() if name.endswith(".pcd")]
            by_class: dict[str, list[str]] = {}
            for name in names:
                parts = Path(name).parts
                obstacle_class = parts[-3]
                by_class.setdefault(obstacle_class, []).append(name)
            for obstacle_class, candidates in sorted(by_class.items()):
                positions = np.linspace(0, len(candidates) - 1, samples, dtype=int)
                for position in positions:
                    name = sorted(candidates)[int(position)]
                    with source.open(name) as stream:
                        values, _ = read_ascii_pcd(stream)
                    rows.append({
                        "archive": archive.name,
                        "source": name,
                        "class": obstacle_class,
                        **evaluate(values, model, False),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("osdar23_root", type=Path)
    parser.add_argument("osdar_ar", nargs="+", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument(
        "--model", type=Path, default=Path("lidar_geometry/models/risk_model.json")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("lidar_geometry/artifacts/osdar_transfer.json"),
    )
    args = parser.parse_args()
    model = RiskModel.load(args.model)
    real = real_rows(args.osdar23_root, model)
    augmented = augmented_rows(args.osdar_ar, model, args.samples_per_class)
    evaluable_targets = [
        row for row in augmented
        if row["target_points"] >= 3 and row["target_in_nominal_clearance"]
    ]
    report = {
        "protocol": {
            "training_on_osdar": False,
            "coordinate_adapter": "OSDaR x/y/z -> MetroGuard -y/x/z",
            "osdar23_sensor": "Pandar64 sensor_index 0 only",
            "osdar_ar_sensor": "merged cloud as published",
            "camera_used": False,
        },
        "real_summary": {
            "frames": len(real),
            "observable_frames": sum(row["observability"] >= 0.3 for row in real),
            "ranked_obstacle_frames": sum(row["ranked_state"] == "OBSTACLE" for row in real),
        },
        "augmented_summary": {
            "frames": len(augmented),
            "target_observable_frames": sum(row["target_points"] >= 3 for row in augmented),
            "target_in_nominal_clearance_frames": sum(
                row["target_in_nominal_clearance"] for row in augmented
            ),
            "geometric_target_frames": sum(row["geometric_target"] is True for row in augmented),
            "ranked_target_frames": sum(row["ranked_target"] is True for row in augmented),
            "evaluable_clearance_targets": len(evaluable_targets),
            "geometric_clearance_target_frames": sum(
                row["geometric_target"] is True for row in evaluable_targets
            ),
            "ranked_clearance_target_frames": sum(
                row["ranked_target"] is True for row in evaluable_targets
            ),
        },
        "real_frames": real,
        "augmented_frames": augmented,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "real_summary": report["real_summary"],
        "augmented_summary": report["augmented_summary"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
