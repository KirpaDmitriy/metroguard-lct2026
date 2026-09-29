"""Ray-consistent synthetic obstacle insertion for PointCloud2 frames.

The generator reuses the real sensor rays already present in a frame.  A ray
that intersects the synthetic primitive has its farther tunnel return replaced
by the nearer obstacle return.  This preserves the Pandar128 firing pattern,
range-dependent sparsity, the real background and first-order occlusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.fast_detector import (
    _NUMPY_FORMATS,
    _fast_track_profile,
    _interpolate_track,
    cloud_arrays,
)
from lidar_geometry.pointcloud2 import PointCloud2


@dataclass(frozen=True)
class SyntheticObject:
    kind: str
    distance_m: float
    lateral_m: float
    width_m: float
    length_m: float
    height_m: float
    bottom_m: float = 0.0
    reflectivity: float = 45.0


@dataclass(frozen=True)
class InjectionResult:
    cloud: PointCloud2
    modified_indices: tuple[int, ...]
    object_bounds_xyz: tuple[tuple[float, float, float], tuple[float, float, float]]
    track_center_m: float
    rail_z_m: float

    @property
    def visible_points(self) -> int:
        return len(self.modified_indices)


def _structured_dtype(cloud: PointCloud2) -> np.dtype:
    names = []
    formats = []
    offsets = []
    endian = ">" if cloud.is_bigendian else "<"
    for field in cloud.fields:
        if field.count != 1 or field.datatype not in _NUMPY_FORMATS:
            continue
        names.append(field.name)
        formats.append(endian + _NUMPY_FORMATS[field.datatype])
        offsets.append(field.offset)
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": cloud.point_step,
        }
    )


def inject_box(
    cloud: PointCloud2,
    obstacle: SyntheticObject,
    config: DetectorConfig = DetectorConfig(),
    seed: int = 0,
) -> InjectionResult:
    arrays = cloud_arrays(cloud)
    x = arrays["x"].astype(np.float64, copy=False)
    y = arrays["y"].astype(np.float64, copy=False)
    z = arrays["z"].astype(np.float64, copy=False)
    distance = -y
    search = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & ((x != 0) | (y != 0) | (z != 0))
        & (distance >= config.min_range_m)
        & (distance <= config.max_range_m)
        & (np.abs(x) <= config.path_search_half_width_m)
    )
    profile, _ = _fast_track_profile(distance[search], x[search], z[search], config)
    if not profile:
        raise ValueError("Cannot place object: rail pair is not observable")
    center, rail_z, supported = _interpolate_track(
        np.array([obstacle.distance_m]), profile, config
    )
    if not supported[0]:
        raise ValueError(
            f"Cannot place object at {obstacle.distance_m:.1f} m outside reliable track range"
        )
    raw_center = float(center[0] + obstacle.lateral_m)
    base_z = float(rail_z[0] + obstacle.bottom_m)
    xyz_min = np.array(
        [
            raw_center - obstacle.width_m / 2,
            -(obstacle.distance_m + obstacle.length_m / 2),
            base_z,
        ]
    )
    xyz_max = np.array(
        [
            raw_center + obstacle.width_m / 2,
            -(obstacle.distance_m - obstacle.length_m / 2),
            base_z + obstacle.height_m,
        ]
    )

    xyz = np.column_stack((x, y, z))
    original_range = np.linalg.norm(xyz, axis=1)
    rays = np.zeros_like(xyz)
    valid_ray = search & (original_range > 0)
    rays[valid_ray] = xyz[valid_ray] / original_range[valid_ray, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / rays
        first = xyz_min * inverse
        second = xyz_max * inverse
        enter = np.max(np.minimum(first, second), axis=1)
        leave = np.min(np.maximum(first, second), axis=1)
    hit = (
        valid_ray
        & np.isfinite(enter)
        & (enter > 0)
        & (leave >= enter)
        & (enter < original_range)
    )
    indices = np.flatnonzero(hit)
    if not len(indices):
        copied = bytearray(cloud.data)
        return InjectionResult(
            PointCloud2(
                cloud.stamp_sec,
                cloud.stamp_nanosec,
                cloud.frame_id,
                cloud.height,
                cloud.width,
                cloud.fields,
                cloud.is_bigendian,
                cloud.point_step,
                cloud.row_step,
                memoryview(copied),
                cloud.is_dense,
            ),
            (),
            (tuple(xyz_min), tuple(xyz_max)),
            float(center[0]),
            float(rail_z[0]),
        )

    copied = bytearray(cloud.data)
    view = np.frombuffer(
        copied, dtype=_structured_dtype(cloud), count=cloud.point_count
    )
    hit_xyz = rays[indices] * enter[indices, None]
    view["x"][indices] = hit_xyz[:, 0]
    view["y"][indices] = hit_xyz[:, 1]
    view["z"][indices] = hit_xyz[:, 2]
    if "intensity" in (view.dtype.names or ()):
        rng = np.random.default_rng(seed)
        noisy = rng.normal(
            obstacle.reflectivity, 0.12 * max(1.0, obstacle.reflectivity), len(indices)
        )
        view["intensity"][indices] = np.clip(noisy, 0, 255)
    injected = PointCloud2(
        cloud.stamp_sec,
        cloud.stamp_nanosec,
        cloud.frame_id,
        cloud.height,
        cloud.width,
        cloud.fields,
        cloud.is_bigendian,
        cloud.point_step,
        cloud.row_step,
        memoryview(copied),
        cloud.is_dense,
    )
    return InjectionResult(
        injected,
        tuple(int(i) for i in indices),
        (tuple(float(v) for v in xyz_min), tuple(float(v) for v in xyz_max)),
        float(center[0]),
        float(rail_z[0]),
    )


SYNTHETIC_LIBRARY = (
    SyntheticObject("minimum_box", 20.0, 0.0, 0.30, 0.30, 0.10, reflectivity=35),
    SyntheticObject("bag", 20.0, 0.0, 0.55, 0.45, 0.35, reflectivity=20),
    SyntheticObject("person", 20.0, 0.0, 0.50, 0.35, 1.70, reflectivity=45),
    SyntheticObject(
        "hanging_cable", 20.0, 0.0, 0.08, 0.08, 1.50, bottom_m=0.8, reflectivity=15
    ),
)
