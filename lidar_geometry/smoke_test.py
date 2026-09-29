from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from lidar_geometry.ego_motion import EgoMotionEstimator
from lidar_geometry.domain_guard import DomainGuard
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import detect_hybrid
from lidar_geometry.pointcloud2 import PointCloud2, iter_bag_messages
from lidar_geometry.risk_model import RiskModel
from lidar_geometry.temporal import ComponentTracker, WorldComponentTracker


def first_cloud(path: Path):
    try:
        return next(iter_bag_messages(path))[1]
    except StopIteration as error:
        raise ValueError(f"No PointCloud2 messages in {path}") from error


def process(path: Path, model: RiskModel, domain_guard: DomainGuard) -> dict:
    cloud = first_cloud(path)
    geometric = detect_fast(cloud)
    ranked = detect_hybrid(cloud, model, domain_guard=domain_guard)
    range_decision = ComponentTracker().update(ranked)
    motion = EgoMotionEstimator().update(cloud)
    world_decision = WorldComponentTracker().update(ranked, motion.cumulative_m)
    payload = {
        "bag": str(path),
        "point_step": cloud.point_step,
        "fields": [field.name for field in cloud.fields],
        "points": cloud.point_count,
        "geometric": asdict(geometric),
        "ranked": asdict(ranked),
        "range_decision": asdict(range_decision),
        "world_decision": asdict(world_decision),
        "motion": asdict(motion),
    }
    json.dumps(payload, allow_nan=False)
    return payload


def malformed_checks(reference: PointCloud2) -> dict:
    empty = PointCloud2(
        reference.stamp_sec,
        reference.stamp_nanosec,
        reference.frame_id,
        1,
        0,
        reference.fields,
        reference.is_bigendian,
        reference.point_step,
        0,
        memoryview(b""),
        reference.is_dense,
    )
    empty_state = detect_fast(empty).state
    missing_intensity = PointCloud2(
        empty.stamp_sec,
        empty.stamp_nanosec,
        empty.frame_id,
        empty.height,
        empty.width,
        tuple(field for field in empty.fields if field.name != "intensity"),
        empty.is_bigendian,
        empty.point_step,
        empty.row_step,
        empty.data,
        empty.is_dense,
    )
    try:
        detect_fast(missing_intensity)
    except ValueError as error:
        missing_field_error = str(error)
    else:
        raise AssertionError("Missing intensity field did not fail")
    return {
        "empty_cloud_state": empty_state,
        "missing_field_error": missing_field_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyzi-bag", type=Path, required=True)
    parser.add_argument("--xyzirt-bag", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("lidar_geometry/models/risk_model.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/submission_smoke.json"),
    )
    parser.add_argument(
        "--domain-guard",
        type=Path,
        default=Path("lidar_geometry/models/domain_guard.npz"),
    )
    args = parser.parse_args()
    model = RiskModel.load(args.model)
    domain_guard = DomainGuard.load(args.domain_guard)
    xyzi = process(args.xyzi_bag, model, domain_guard)
    xyzirt = process(args.xyzirt_bag, model, domain_guard)
    report = {
        "status": "PASS",
        "formats": [xyzi, xyzirt],
        "malformed_checks": malformed_checks(first_cloud(args.xyzirt_bag)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "point_steps": [item["point_step"] for item in report["formats"]],
        "states": [item["ranked"]["state"] for item in report["formats"]],
        **report["malformed_checks"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
