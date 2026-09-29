"""Geometric obstacle detector for the metro lidar hackathon data.

Coordinate convention in the supplied bags:

* ``-y`` is forward along the track;
* ``x`` is lateral;
* ``z`` is vertical.

The detector estimates the local rail/ground profile, keeps points inside a
configurable swept train envelope, removes the track surface, then joins the
remaining points into 3-D voxel components.  It is a deliberately transparent
baseline, not a claim that a single rectangular corridor solves curved track.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Iterable

from lidar_geometry.pointcloud2 import PointCloud2, is_valid_xyz, iter_bag_messages


@dataclass(frozen=True)
class DetectorConfig:
    min_range_m: float = 3.0
    max_range_m: float = 150.0
    # Q&A 00:24:44 and 00:41:29: 2.1 m wide, 3.0 m high, including protrusions.
    half_width_m: float = 1.05
    component_guard_band_m: float = 0.0
    # Minimum protrusion above the locally estimated track surface.  This is
    # deliberately below the 0.10 m baseline-object height from Q&A 00:23:56.
    min_height_above_rail_m: float = 0.06
    clearance_height_m: float = 3.0
    # Disabled until an official clearance contour is supplied.
    roof_taper_start_m: float | None = None
    roof_half_width_at_top_m: float = 0.9
    rail_gauge_m: float = 1.52
    rail_gauge_tolerance_m: float = 0.18
    rail_pair_max_z_difference_m: float = 0.15
    path_search_half_width_m: float = 4.0
    path_cell_m: float = 0.05
    path_max_cells_per_bin: int = 120
    path_min_cell_neighbourhood: int = 2
    path_initial_center_tolerance_m: float = 0.55
    path_center_step_m: float = 0.30
    path_center_step_per_missing_bin_m: float = 0.12
    rail_head_search_half_width_m: float = 0.14
    rail_head_search_z_m: float = 0.35
    rail_head_top_quantile: float = 0.98
    floor_bin_m: float = 2.0
    floor_quantile: float = 0.20
    # q20 tends to describe ballast/trough; this lifts it to approximate rail top.
    rail_above_floor_quantile_m: float = 0.22
    floor_sample_max_z_m: float = 0.5
    floor_min_points_per_bin: int = 8
    surface_distance_cell_m: float = 0.50
    surface_lateral_cell_m: float = 0.10
    surface_longitudinal_radius_cells: int = 4
    surface_max_height_above_rail_m: float = 0.60
    voxel_m: float = 0.20
    min_cluster_points: int = 8
    min_cluster_voxels: int = 2
    # A 10 cm box may expose only its horizontal top to a roof-mounted lidar;
    # do not require a visible vertical face. Horizontal support is checked.
    min_cluster_vertical_extent_m: float = 0.0
    min_cluster_top_height_m: float = 0.08
    min_cluster_horizontal_extent_m: float = 0.12
    rail_like_max_height_m: float = 0.25
    rail_like_min_length_m: float = 1.0
    rail_like_max_width_m: float = 0.25
    rail_like_lateral_tolerance_m: float = 0.20


@dataclass(frozen=True)
class CandidatePoint:
    distance: float
    lateral: float
    z: float
    height: float
    surface_residual: float
    intensity: float
    ring: int


@dataclass(frozen=True)
class Obstacle:
    points: int
    voxels: int
    distance_min_m: float
    distance_max_m: float
    lateral_min_m: float
    lateral_max_m: float
    height_min_m: float
    height_max_m: float
    surface_residual_max_m: float
    surface_residual_mean_m: float
    intensity_mean: float


@dataclass(frozen=True)
class Detection:
    obstacle: bool
    nearest_distance_m: float | None
    candidates: int
    track_bins: int
    reliable_range_min_m: float | None
    reliable_range_max_m: float | None
    obstacles: tuple[Obstacle, ...]


@dataclass(frozen=True)
class TrackSample:
    center: float
    rail_z: float
    gauge: float
    score: float


def _quantile(values: list[float], fraction: float) -> float:
    values.sort()
    index = round(fraction * (len(values) - 1))
    return values[index]


def _estimate_rail_profile(
    points: Iterable[tuple[float, float, float, float, int]], config: DetectorConfig
) -> dict[int, float]:
    samples: dict[int, list[float]] = {}
    for distance, lateral, z, _intensity, _ring in points:
        if z > config.floor_sample_max_z_m:
            continue
        index = int(distance // config.floor_bin_m)
        samples.setdefault(index, []).append(z)

    raw = {
        index: _quantile(values, config.floor_quantile)
        + config.rail_above_floor_quantile_m
        for index, values in samples.items()
        if len(values) >= config.floor_min_points_per_bin
    }
    if not raw:
        return {}

    # A median filter rejects a bin raised by an object hiding part of the floor.
    smoothed: dict[int, float] = {}
    indexes = sorted(raw)
    for index in indexes:
        neighbours = [raw[j] for j in range(index - 2, index + 3) if j in raw]
        smoothed[index] = statistics.median(neighbours)

    # Fill gaps by linear interpolation. Outside the observed interval, retain
    # the closest estimate; sparse far-range detections are reported cautiously.
    first, last = indexes[0], indexes[-1]
    for index in range(first, last + 1):
        if index in smoothed:
            continue
        left = max((j for j in indexes if j < index), default=first)
        right = min((j for j in indexes if j > index), default=last)
        if left == right:
            smoothed[index] = smoothed[left]
        else:
            alpha = (index - left) / (right - left)
            smoothed[index] = smoothed[left] * (1 - alpha) + smoothed[right] * alpha
    return smoothed


def _rail_z(distance: float, profile: dict[int, float], config: DetectorConfig) -> float | None:
    if not profile:
        return None
    position = distance / config.floor_bin_m
    left = math.floor(position)
    right = left + 1
    minimum = min(profile)
    maximum = max(profile)
    left = min(max(left, minimum), maximum)
    right = min(max(right, minimum), maximum)
    if left == right:
        return profile[left]
    alpha = position - math.floor(position)
    return profile[left] * (1 - alpha) + profile[right] * alpha


def _estimate_track_profile(
    points: Iterable[tuple[float, float, float, float, int]], config: DetectorConfig
) -> dict[int, TrackSample]:
    """Find two approximately gauge-separated rail ridges in consecutive bins."""
    cell = config.path_cell_m
    bins: dict[int, Counter[tuple[int, int]]] = {}
    raw_bins: dict[int, list[tuple[float, float]]] = {}
    for distance, lateral, z, _intensity, _ring in points:
        if z > config.floor_sample_max_z_m:
            continue
        index = int(distance // config.floor_bin_m)
        key = (round(lateral / cell), round(z / cell))
        bins.setdefault(index, Counter())[key] += 1
        raw_bins.setdefault(index, []).append((lateral, z))

    selected: dict[int, TrackSample] = {}
    predicted_center = 0.0
    previous_index: int | None = None
    for index in sorted(bins):
        histogram = bins[index]
        ranked = []
        for (ix, iz), _count in histogram.items():
            neighbourhood = sum(
                histogram.get((ix + dx, iz + dz), 0)
                for dx in (-1, 0, 1)
                for dz in (-1, 0, 1)
            )
            if neighbourhood >= config.path_min_cell_neighbourhood:
                ranked.append((neighbourhood, ix, iz))
        ranked.sort(reverse=True)
        ranked = ranked[: config.path_max_cells_per_bin]

        missing_bins = 0 if previous_index is None else index - previous_index
        allowed_shift = (
            config.path_initial_center_tolerance_m
            if previous_index is None
            else config.path_center_step_m
            + config.path_center_step_per_missing_bin_m * missing_bins
        )
        best: TrackSample | None = None
        for left_count, left_x, left_z in ranked:
            for right_count, right_x, right_z in ranked:
                if right_x <= left_x:
                    continue
                gauge = (right_x - left_x) * cell
                z_difference = abs(right_z - left_z) * cell
                center = (left_x + right_x) * cell / 2
                if abs(gauge - config.rail_gauge_m) > config.rail_gauge_tolerance_m:
                    continue
                if z_difference > config.rail_pair_max_z_difference_m:
                    continue
                if abs(center - predicted_center) > allowed_shift:
                    continue
                score = (
                    left_count
                    + right_count
                    - 10 * abs(gauge - config.rail_gauge_m)
                    - 8 * z_difference
                    - 10 * abs(center - predicted_center)
                )
                sample = TrackSample(
                    center=center,
                    rail_z=(left_z + right_z) * cell / 2,
                    gauge=gauge,
                    score=score,
                )
                if best is None or sample.score > best.score:
                    best = sample
        if best is not None:
            # The dense cell pair locates the rail ridges, but its vertical
            # center is usually below the actual head surface. Refine z from
            # the upper returns near both heads. Taking the lower of the two
            # estimates resists an object sitting on one rail.
            head_tops = []
            for expected_x in (
                best.center - best.gauge / 2,
                best.center + best.gauge / 2,
            ):
                head_points = [
                    z
                    for x, z in raw_bins[index]
                    if abs(x - expected_x) <= config.rail_head_search_half_width_m
                    and abs(z - best.rail_z) <= config.rail_head_search_z_m
                ]
                if len(head_points) >= 3:
                    head_tops.append(_quantile(head_points, config.rail_head_top_quantile))
            if head_tops:
                best = TrackSample(
                    center=best.center,
                    rail_z=min(head_tops),
                    gauge=best.gauge,
                    score=best.score,
                )
            selected[index] = best
            predicted_center = best.center
            previous_index = index

    if not selected:
        return {}

    # Median smoothing preserves a bend while rejecting a one-bin jump to a
    # platform edge or tunnel fixture that happens to be gauge-separated.
    smoothed: dict[int, TrackSample] = {}
    indexes = sorted(selected)
    for index in indexes:
        neighbours = [selected[j] for j in range(index - 2, index + 3) if j in selected]
        current = selected[index]
        smoothed[index] = TrackSample(
            center=statistics.median(item.center for item in neighbours),
            rail_z=statistics.median(item.rail_z for item in neighbours),
            gauge=current.gauge,
            score=current.score,
        )

    # Interpolate unobserved bins. Beyond the last reliable rail pair we keep
    # the last estimate instead of inventing a curve from sparse points.
    first, last = indexes[0], indexes[-1]
    for index in range(first, last + 1):
        if index in smoothed:
            continue
        left = max((j for j in indexes if j < index), default=first)
        right = min((j for j in indexes if j > index), default=last)
        if left == right:
            smoothed[index] = smoothed[left]
            continue
        alpha = (index - left) / (right - left)
        smoothed[index] = TrackSample(
            center=smoothed[left].center * (1 - alpha) + smoothed[right].center * alpha,
            rail_z=smoothed[left].rail_z * (1 - alpha) + smoothed[right].rail_z * alpha,
            gauge=config.rail_gauge_m,
            score=min(smoothed[left].score, smoothed[right].score),
        )
    return smoothed


def _track_at(
    distance: float, profile: dict[int, TrackSample], config: DetectorConfig
) -> TrackSample | None:
    if not profile:
        return None
    position = distance / config.floor_bin_m
    minimum = min(profile)
    maximum = max(profile)
    # Rail returns disappear before walls and large obstacles do. Extrapolate
    # the recent path trend instead of silently turning a curved track into a
    # straight corridor at the last observed bin.
    if position > maximum and len(profile) >= 2:
        tail = [profile[index] for index in sorted(profile)[-5:]]
        center_steps = [b.center - a.center for a, b in zip(tail, tail[1:])]
        z_steps = [b.rail_z - a.rail_z for a, b in zip(tail, tail[1:])]
        center_step = max(
            -config.path_center_step_m,
            min(config.path_center_step_m, statistics.median(center_steps)),
        )
        z_step = statistics.median(z_steps)
        delta = position - maximum
        last = profile[maximum]
        return TrackSample(
            center=last.center + center_step * delta,
            rail_z=last.rail_z + z_step * delta,
            gauge=config.rail_gauge_m,
            score=last.score / (1 + delta),
        )
    left = min(max(math.floor(position), minimum), maximum)
    right = min(max(left + 1, minimum), maximum)
    if left == right:
        return profile[left]
    alpha = position - math.floor(position)
    return TrackSample(
        center=profile[left].center * (1 - alpha) + profile[right].center * alpha,
        rail_z=profile[left].rail_z * (1 - alpha) + profile[right].rail_z * alpha,
        gauge=config.rail_gauge_m,
        score=min(profile[left].score, profile[right].score),
    )


def _estimate_track_surface(
    points: Iterable[tuple[float, float, float, float]], config: DetectorConfig
) -> dict[tuple[int, int], float]:
    """Estimate the normal 2.5-D path surface in track-relative coordinates.

    A height threshold relative only to the rail plane leaves rail heads,
    sleepers and fasteners as obstacles.  Here each lateral strip is smoothed
    longitudinally: structures continuing along the track become background,
    while a compact object remains a positive residual.  The current cell is
    included, but the wide median window prevents a short object from lifting
    its own reference surface.
    """
    raw_samples: dict[tuple[int, int], list[float]] = {}
    for distance, lateral, z, rail_z in points:
        height = z - rail_z
        if not (-0.50 <= height <= config.surface_max_height_above_rail_m):
            continue
        key = (
            math.floor(distance / config.surface_distance_cell_m),
            math.floor(lateral / config.surface_lateral_cell_m),
        )
        raw_samples.setdefault(key, []).append(z)

    # A median inside a cell rejects isolated multipath returns.
    raw = {key: statistics.median(values) for key, values in raw_samples.items()}
    surface: dict[tuple[int, int], float] = {}
    radius = config.surface_longitudinal_radius_cells
    for distance_index, lateral_index in raw:
        neighbours = [
            raw[(other_distance, lateral_index)]
            for other_distance in range(distance_index - radius, distance_index + radius + 1)
            if (other_distance, lateral_index) in raw
        ]
        if neighbours:
            surface[(distance_index, lateral_index)] = statistics.median(neighbours)
    return surface


def _surface_z(
    distance: float,
    lateral: float,
    surface: dict[tuple[int, int], float],
    fallback: float,
    config: DetectorConfig,
) -> float:
    distance_index = math.floor(distance / config.surface_distance_cell_m)
    lateral_index = math.floor(lateral / config.surface_lateral_cell_m)
    exact = surface.get((distance_index, lateral_index))
    if exact is not None:
        return exact
    # Sparse long-range returns often miss an exact cell.  Only borrow from a
    # very small neighbourhood so a wall/platform cannot define the track bed.
    nearby = [
        surface[(di, li)]
        for di in range(distance_index - 1, distance_index + 2)
        for li in range(lateral_index - 1, lateral_index + 2)
        if (di, li) in surface
    ]
    return statistics.median(nearby) if nearby else fallback


def _candidate_points(cloud: PointCloud2, config: DetectorConfig):
    search_points: list[tuple[float, float, float, float, int]] = []
    for x, y, z, intensity, ring, _point_timestamp in cloud.iter_xyzirt():
        if not is_valid_xyz(x, y, z):
            continue
        distance = -y
        if not (config.min_range_m <= distance <= config.max_range_m):
            continue
        if abs(x) > config.path_search_half_width_m:
            continue
        search_points.append((distance, x, z, intensity, ring))

    track_profile = _estimate_track_profile(search_points, config)
    # Fallback keeps the detector usable when two rail ridges cannot be found.
    fallback_points = [point for point in search_points if abs(point[1]) <= config.half_width_m]
    fallback_rail_profile = _estimate_rail_profile(fallback_points, config)
    aligned_points: list[tuple[float, float, float, float, int, float]] = []
    reliable_min_distance = (
        min(track_profile) * config.floor_bin_m if track_profile else config.min_range_m
    )
    reliable_max_distance = (
        (max(track_profile) + 1) * config.floor_bin_m
        if track_profile
        else config.max_range_m
    )
    for distance, raw_lateral, z, intensity, ring in search_points:
        # Before the first visible rail return this sensor sees parts of the
        # train itself. Beyond the last rail-supported bin the swept corridor
        # is unobservable and a wall on a bend is easily mistaken for cargo.
        if not (reliable_min_distance <= distance <= reliable_max_distance):
            continue
        track = _track_at(distance, track_profile, config)
        center = track.center if track is not None else 0.0
        rail_z = track.rail_z if track is not None else _rail_z(
            distance, fallback_rail_profile, config
        )
        if rail_z is None:
            continue
        lateral = raw_lateral - center
        if abs(lateral) > config.half_width_m:
            continue
        aligned_points.append((distance, lateral, z, intensity, ring, rail_z))

    surface = _estimate_track_surface(
        ((distance, lateral, z, rail_z) for distance, lateral, z, _, _, rail_z in aligned_points),
        config,
    )
    candidates: list[CandidatePoint] = []
    for distance, lateral, z, intensity, ring, rail_z in aligned_points:
        height = z - rail_z
        if not (-0.05 <= height <= config.clearance_height_m):
            continue
        surface_residual = z - _surface_z(distance, lateral, surface, rail_z, config)
        if surface_residual < config.min_height_above_rail_m:
            continue
        if config.roof_taper_start_m is not None and height > config.roof_taper_start_m:
            roof_fraction = (height - config.roof_taper_start_m) / (
                config.clearance_height_m - config.roof_taper_start_m
            )
            allowed_half_width = config.half_width_m * (1 - roof_fraction) + (
                config.roof_half_width_at_top_m * roof_fraction
            )
            if abs(lateral) > allowed_half_width:
                continue
        candidates.append(
            CandidatePoint(distance, lateral, z, height, surface_residual, intensity, ring)
        )
    return candidates, track_profile, fallback_rail_profile


def _cluster(points: list[CandidatePoint], config: DetectorConfig) -> tuple[Obstacle, ...]:
    cells: dict[tuple[int, int, int], list[CandidatePoint]] = {}
    size = config.voxel_m
    for point in points:
        key = (
            math.floor(point.distance / size),
            math.floor(point.lateral / size),
            math.floor(point.height / size),
        )
        cells.setdefault(key, []).append(point)

    remaining = set(cells)
    obstacles: list[Obstacle] = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        component_keys = [seed]
        while queue:
            current = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbour = (current[0] + dx, current[1] + dy, current[2] + dz)
                        if neighbour in remaining:
                            remaining.remove(neighbour)
                            queue.append(neighbour)
                            component_keys.append(neighbour)

        component = [point for key in component_keys for point in cells[key]]
        if len(component) < config.min_cluster_points:
            continue
        if len(component_keys) < config.min_cluster_voxels:
            continue
        height_min = min(point.height for point in component)
        height_max = max(point.height for point in component)
        if height_max < config.min_cluster_top_height_m:
            continue
        if height_max - height_min < config.min_cluster_vertical_extent_m:
            continue
        lateral_min = min(point.lateral for point in component)
        lateral_max = max(point.lateral for point in component)
        distance_min = min(point.distance for point in component)
        distance_max = max(point.distance for point in component)
        if max(distance_max - distance_min, lateral_max - lateral_min) < (
            config.min_cluster_horizontal_extent_m
        ):
            continue
        lateral_center = (lateral_min + lateral_max) / 2
        # Rail returns form low, narrow components extending along the path.
        # Do this at component level so a compact 0.30 x 0.30 x 0.10 m object
        # placed on a rail is not erased together with the rail points.
        if (
            height_max <= config.rail_like_max_height_m
            and distance_max - distance_min >= config.rail_like_min_length_m
            and lateral_max - lateral_min <= config.rail_like_max_width_m
            and abs(abs(lateral_center) - config.rail_gauge_m / 2)
            <= config.rail_like_lateral_tolerance_m
        ):
            continue
        obstacles.append(
            Obstacle(
                points=len(component),
                voxels=len(component_keys),
                distance_min_m=distance_min,
                distance_max_m=distance_max,
                lateral_min_m=lateral_min,
                lateral_max_m=lateral_max,
                height_min_m=height_min,
                height_max_m=height_max,
                surface_residual_max_m=max(point.surface_residual for point in component),
                surface_residual_mean_m=(
                    sum(point.surface_residual for point in component) / len(component)
                ),
                intensity_mean=sum(point.intensity for point in component) / len(component),
            )
        )
    obstacles.sort(key=lambda item: item.distance_min_m)
    return tuple(obstacles)


def detect(cloud: PointCloud2, config: DetectorConfig = DetectorConfig()) -> Detection:
    candidates, track_profile, fallback_rail_profile = _candidate_points(cloud, config)
    obstacles = _cluster(candidates, config)
    active_profile = track_profile if track_profile else fallback_rail_profile
    return Detection(
        obstacle=bool(obstacles),
        nearest_distance_m=obstacles[0].distance_min_m if obstacles else None,
        candidates=len(candidates),
        track_bins=len(active_profile),
        reliable_range_min_m=(
            min(active_profile) * config.floor_bin_m if active_profile else None
        ),
        reliable_range_max_m=(
            (max(active_profile) + 1) * config.floor_bin_m if active_profile else None
        ),
        obstacles=obstacles,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="Path to a rosbag2 .db3 file")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--half-width", type=float, default=DetectorConfig.half_width_m)
    parser.add_argument("--min-range", type=float, default=DetectorConfig.min_range_m)
    parser.add_argument("--max-range", type=float, default=DetectorConfig.max_range_m)
    parser.add_argument("--min-height", type=float, default=DetectorConfig.min_height_above_rail_m)
    parser.add_argument("--clearance-height", type=float, default=DetectorConfig.clearance_height_m)
    parser.add_argument("--min-points", type=int, default=DetectorConfig.min_cluster_points)
    parser.add_argument("--voxel", type=float, default=DetectorConfig.voxel_m)
    parser.add_argument("--show-clusters", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = DetectorConfig(
        half_width_m=args.half_width,
        min_range_m=args.min_range,
        max_range_m=args.max_range,
        min_height_above_rail_m=args.min_height,
        clearance_height_m=args.clearance_height,
        min_cluster_points=args.min_points,
        voxel_m=args.voxel,
    )
    processed = 0
    started = time.perf_counter()
    for source_index, (bag_timestamp, cloud) in enumerate(iter_bag_messages(args.bag)):
        if source_index % args.every:
            continue
        result = detect(cloud, config)
        output = {
            "frame": source_index,
            "bag_timestamp_ns": bag_timestamp,
            "obstacle": result.obstacle,
            "nearest_distance_m": result.nearest_distance_m,
            "candidate_points": result.candidates,
            "track_bins": result.track_bins,
            "reliable_range_m": [
                result.reliable_range_min_m,
                result.reliable_range_max_m,
            ],
            "clusters": [asdict(item) for item in result.obstacles[: args.show_clusters]],
        }
        print(json.dumps(output, ensure_ascii=False))
        processed += 1
        if args.max_frames is not None and processed >= args.max_frames:
            break
    elapsed = time.perf_counter() - started
    rate = processed / elapsed if elapsed else 0.0
    print(
        json.dumps(
            {"summary": {"frames": processed, "elapsed_s": elapsed, "fps": rate}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
