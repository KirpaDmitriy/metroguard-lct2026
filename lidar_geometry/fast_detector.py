"""Vectorized, safety-state obstacle detector for the metro LiDAR task.

The original :mod:`detect_obstacles` module is the readable reference model.
This module keeps the same geometric idea, but moves the expensive point-wise
operations to NumPy and adds an explicit UNKNOWN state.  It intentionally does
not classify obstacle type: the task is intrusion into the swept train volume.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from lidar_geometry.detect_obstacles import (
    CandidatePoint,
    DetectorConfig,
    Obstacle,
    TrackSample,
    _cluster,
)
from lidar_geometry.pointcloud2 import PointCloud2, iter_bag_messages

_NUMPY_FORMATS = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}


@dataclass(frozen=True)
class SafetyDetection:
    state: str
    obstacle: bool
    nearest_distance_m: float | None
    confidence: float
    observability: float
    candidate_points: int
    track_bins: int
    observed_track_bins: int
    reliable_range_min_m: float | None
    reliable_range_max_m: float | None
    obstacles: tuple[Obstacle, ...]
    reason: str


CORE_HALF_WIDTH_M = 0.60
TrackModel = Literal["greedy", "continuity"]


def cloud_arrays(cloud: PointCloud2) -> dict[str, np.ndarray]:
    """Expose PointCloud2 fields as NumPy arrays.

    The original bags contain a 26-byte point with ``ring`` and ``timestamp``.
    The organisers' synthetic bag currently contains a 16-byte XYZI point.
    Geometry does not require a laser-ring id, so missing rings are represented
    by ``-1`` instead of rejecting an otherwise valid cloud.
    """
    fields = {field.name: field for field in cloud.fields}
    required = ("x", "y", "z", "intensity")
    missing = set(required) - set(fields)
    if missing:
        raise ValueError(f"PointCloud2 fields missing: {sorted(missing)}")
    selected = required + (("ring",) if "ring" in fields else ())
    endian = ">" if cloud.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": list(selected),
            "formats": [
                endian + _NUMPY_FORMATS[fields[name].datatype] for name in selected
            ],
            "offsets": [fields[name].offset for name in selected],
            "itemsize": cloud.point_step,
        }
    )
    view = np.frombuffer(cloud.data, dtype=dtype, count=cloud.point_count)
    arrays = {name: view[name] for name in required}
    arrays["ring"] = (
        view["ring"]
        if "ring" in fields
        else np.full(cloud.point_count, -1, dtype=np.int32)
    )
    return arrays


def _top_quantile(values: np.ndarray, fraction: float) -> float:
    if not len(values):
        return math.nan
    return float(np.quantile(values, fraction, method="nearest"))


def _neighbourhood_3x3(histogram: np.ndarray) -> np.ndarray:
    padded = np.pad(histogram, 1)
    vertical = padded[:-2] + padded[1:-1] + padded[2:]
    return vertical[:, :-2] + vertical[:, 1:-1] + vertical[:, 2:]


def _nanmedian_last_axis(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=-1)
    ordered = np.sort(np.where(finite, values, np.inf), axis=-1)
    lower_index = np.maximum(0, (count - 1) // 2)
    upper_index = np.maximum(0, count // 2)
    lower = np.take_along_axis(ordered, lower_index[..., None], axis=-1)[..., 0]
    upper = np.take_along_axis(ordered, upper_index[..., None], axis=-1)[..., 0]
    median = (lower + upper) / 2
    return np.where(count > 0, median, np.nan)


def _greedy_track_profile(
    distance: np.ndarray,
    lateral: np.ndarray,
    z: np.ndarray,
    config: DetectorConfig,
) -> tuple[dict[int, TrackSample], int]:
    """Find gauge-separated ridges using per-range 2-D occupancy grids."""
    low = z <= config.floor_sample_max_z_m
    if not np.any(low):
        return {}, 0
    d = distance[low]
    x = lateral[low]
    zz = z[low]
    db = np.floor(d / config.floor_bin_m).astype(np.int32)
    order = np.argsort(db)
    db, d, x, zz = db[order], d[order], x[order], zz[order]
    split = np.flatnonzero(np.diff(db)) + 1
    ranges = np.split(np.arange(len(db)), split)

    cell = config.path_cell_m
    x_min = -config.path_search_half_width_m
    x_count = int(round(2 * config.path_search_half_width_m / cell)) + 1
    z_min = -5.0
    z_count = int(round((config.floor_sample_max_z_m - z_min) / cell)) + 1

    selected: dict[int, TrackSample] = {}
    predicted_center = 0.0
    previous_index: int | None = None
    for indexes in ranges:
        index = int(db[indexes[0]])
        xx = x[indexes]
        zvals = zz[indexes]
        ix = np.rint((xx - x_min) / cell).astype(np.int32)
        iz = np.rint((zvals - z_min) / cell).astype(np.int32)
        valid = (ix >= 0) & (ix < x_count) & (iz >= 0) & (iz < z_count)
        if np.count_nonzero(valid) < config.floor_min_points_per_bin:
            continue
        flat = ix[valid] * z_count + iz[valid]
        histogram = np.bincount(flat, minlength=x_count * z_count).reshape(
            x_count, z_count
        )
        neighbourhood = _neighbourhood_3x3(histogram)
        flat_scores = neighbourhood.ravel()
        eligible = np.flatnonzero(flat_scores >= config.path_min_cell_neighbourhood)
        if not len(eligible):
            continue
        keep = min(config.path_max_cells_per_bin, len(eligible))
        if keep < len(eligible):
            choice = eligible[np.argpartition(flat_scores[eligible], -keep)[-keep:]]
        else:
            choice = eligible
        scores = flat_scores[choice].astype(np.float64)
        cx = choice // z_count
        cz = choice % z_count
        candidate_x = x_min + cx * cell
        candidate_z = z_min + cz * cell

        missing_bins = 0 if previous_index is None else index - previous_index
        allowed_shift = (
            config.path_initial_center_tolerance_m
            if previous_index is None
            else config.path_center_step_m
            + config.path_center_step_per_missing_bin_m * missing_bins
        )
        gauge = candidate_x[None, :] - candidate_x[:, None]
        z_difference = np.abs(candidate_z[None, :] - candidate_z[:, None])
        center = (candidate_x[None, :] + candidate_x[:, None]) / 2
        pair_score = (
            scores[None, :]
            + scores[:, None]
            - 10 * np.abs(gauge - config.rail_gauge_m)
            - 8 * z_difference
            - 10 * np.abs(center - predicted_center)
        )
        valid_pair = (
            (gauge > 0)
            & (np.abs(gauge - config.rail_gauge_m) <= config.rail_gauge_tolerance_m)
            & (z_difference <= config.rail_pair_max_z_difference_m)
            & (np.abs(center - predicted_center) <= allowed_shift)
        )
        if not np.any(valid_pair):
            continue
        pair_score[~valid_pair] = -np.inf
        left_pos, right_pos = np.unravel_index(np.argmax(pair_score), pair_score.shape)
        best = TrackSample(
            center=float(center[left_pos, right_pos]),
            rail_z=float((candidate_z[left_pos] + candidate_z[right_pos]) / 2),
            gauge=float(gauge[left_pos, right_pos]),
            score=float(pair_score[left_pos, right_pos]),
        )
        head_tops = []
        for expected_x in (best.center - best.gauge / 2, best.center + best.gauge / 2):
            head = zvals[
                (np.abs(xx - expected_x) <= config.rail_head_search_half_width_m)
                & (np.abs(zvals - best.rail_z) <= config.rail_head_search_z_m)
            ]
            if len(head) >= 3:
                head_tops.append(_top_quantile(head, config.rail_head_top_quantile))
        if head_tops:
            best = TrackSample(best.center, min(head_tops), best.gauge, best.score)
        selected[index] = best
        predicted_center = best.center
        previous_index = index

    observed_count = len(selected)
    if not selected:
        return {}, 0
    indexes = sorted(selected)
    smoothed: dict[int, TrackSample] = {}
    for index in indexes:
        neighbours = [selected[j] for j in range(index - 2, index + 3) if j in selected]
        current = selected[index]
        smoothed[index] = TrackSample(
            statistics.median(item.center for item in neighbours),
            statistics.median(item.rail_z for item in neighbours),
            current.gauge,
            current.score,
        )
    first, last = indexes[0], indexes[-1]
    known = np.array(indexes)
    known_center = np.array([smoothed[i].center for i in indexes])
    known_z = np.array([smoothed[i].rail_z for i in indexes])
    full = np.arange(first, last + 1)
    centers = np.interp(full, known, known_center)
    rail_z = np.interp(full, known, known_z)
    for pos, index in enumerate(full):
        if int(index) not in smoothed:
            smoothed[int(index)] = TrackSample(
                float(centers[pos]), float(rail_z[pos]), config.rail_gauge_m, 0.0
            )
    return smoothed, observed_count


def _continuity_track_profile(
    distance: np.ndarray,
    lateral: np.ndarray,
    z: np.ndarray,
    config: DetectorConfig,
) -> tuple[dict[int, TrackSample], int] | None:
    low = z <= config.floor_sample_max_z_m
    if not np.any(low):
        return None
    d = distance[low]
    x = lateral[low]
    zz = z[low]
    db = np.floor(d / config.floor_bin_m).astype(np.int32)
    order = np.argsort(db)
    db, x, zz = db[order], x[order], zz[order]
    split = np.flatnonzero(np.diff(db)) + 1

    cell = config.path_cell_m
    x_min = -config.path_search_half_width_m
    x_count = int(round(2 * config.path_search_half_width_m / cell)) + 1
    z_min = -5.0
    z_count = int(round((config.floor_sample_max_z_m - z_min) / cell)) + 1

    observations: list[tuple[int, list[TrackSample], np.ndarray, np.ndarray]] = []
    for indexes in np.split(np.arange(len(db)), split):
        index = int(db[indexes[0]])
        xx = x[indexes]
        zvals = zz[indexes]
        ix = np.rint((xx - x_min) / cell).astype(np.int32)
        iz = np.rint((zvals - z_min) / cell).astype(np.int32)
        valid = (ix >= 0) & (ix < x_count) & (iz >= 0) & (iz < z_count)
        if np.count_nonzero(valid) < config.floor_min_points_per_bin:
            continue
        flat = ix[valid] * z_count + iz[valid]
        histogram = np.bincount(flat, minlength=x_count * z_count).reshape(
            x_count, z_count
        )
        neighbourhood = _neighbourhood_3x3(histogram)
        flat_scores = neighbourhood.ravel()
        eligible = np.flatnonzero(flat_scores >= config.path_min_cell_neighbourhood)
        if not len(eligible):
            continue
        keep = min(config.path_max_cells_per_bin, len(eligible))
        if keep < len(eligible):
            choice = eligible[np.argpartition(flat_scores[eligible], -keep)[-keep:]]
        else:
            choice = eligible
        scores = flat_scores[choice].astype(np.float64)
        cx = choice // z_count
        cz = choice % z_count
        candidate_x = x_min + cx * cell
        candidate_z = z_min + cz * cell
        gauge = candidate_x[None, :] - candidate_x[:, None]
        z_difference = np.abs(candidate_z[None, :] - candidate_z[:, None])
        center = (candidate_x[None, :] + candidate_x[:, None]) / 2
        emission = (
            scores[None, :]
            + scores[:, None]
            - 10 * np.abs(gauge - config.rail_gauge_m)
            - 8 * z_difference
        )
        valid_pair = (
            (gauge > 0)
            & (np.abs(gauge - config.rail_gauge_m) <= config.rail_gauge_tolerance_m)
            & (z_difference <= config.rail_pair_max_z_difference_m)
        )
        pair_positions = np.argwhere(valid_pair)
        if not len(pair_positions):
            continue
        pair_values = emission[valid_pair]
        pair_keep = min(16, len(pair_values))
        if pair_keep < len(pair_values):
            top = np.argpartition(pair_values, -pair_keep)[-pair_keep:]
            pair_positions = pair_positions[top]
            pair_values = pair_values[top]
        candidates = [
            TrackSample(
                center=float(center[left, right]),
                rail_z=float((candidate_z[left] + candidate_z[right]) / 2),
                gauge=float(gauge[left, right]),
                score=float(value),
            )
            for (left, right), value in zip(pair_positions, pair_values)
        ]
        observations.append((index, candidates, xx, zvals))

    if not observations:
        return None
    first_candidates = observations[0][1]
    first_centers = np.array([item.center for item in first_candidates])
    allowed_first = np.abs(first_centers) <= config.path_initial_center_tolerance_m
    scores = np.array([item.score for item in first_candidates]) - 10 * np.abs(
        first_centers
    )
    scores[~allowed_first] = -np.inf
    if not np.any(np.isfinite(scores)):
        return None
    score_layers = [scores]
    parents: list[np.ndarray] = []
    for position in range(1, len(observations)):
        previous_index, previous, _xx, _zz = observations[position - 1]
        index, current, _xx, _zz = observations[position]
        previous_center = np.array([item.center for item in previous])
        previous_z = np.array([item.rail_z for item in previous])
        current_center = np.array([item.center for item in current])
        current_z = np.array([item.rail_z for item in current])
        center_step = np.abs(previous_center[:, None] - current_center[None, :])
        z_step = np.abs(previous_z[:, None] - current_z[None, :])
        allowed_shift = (
            config.path_center_step_m
            + config.path_center_step_per_missing_bin_m * (index - previous_index)
        )
        transitions = score_layers[-1][:, None] - 10 * center_step - 8 * z_step
        transitions[center_step > allowed_shift] = -np.inf
        parent = np.argmax(transitions, axis=0)
        best = transitions[parent, np.arange(len(current))]
        next_scores = best + np.array([item.score for item in current])
        if not np.any(np.isfinite(next_scores)):
            return None
        parents.append(parent)
        score_layers.append(next_scores)

    path = [int(np.argmax(score_layers[-1]))]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    path.reverse()

    selected: dict[int, TrackSample] = {}
    for (index, candidates, xx, zvals), choice in zip(observations, path):
        best = candidates[choice]
        head_tops = []
        for expected_x in (best.center - best.gauge / 2, best.center + best.gauge / 2):
            head = zvals[
                (np.abs(xx - expected_x) <= config.rail_head_search_half_width_m)
                & (np.abs(zvals - best.rail_z) <= config.rail_head_search_z_m)
            ]
            if len(head) >= 3:
                head_tops.append(_top_quantile(head, config.rail_head_top_quantile))
        if head_tops:
            best = TrackSample(best.center, min(head_tops), best.gauge, best.score)
        selected[index] = best

    observed_count = len(selected)
    indexes = sorted(selected)
    smoothed = {}
    for index in indexes:
        neighbours = [selected[j] for j in range(index - 2, index + 3) if j in selected]
        current = selected[index]
        smoothed[index] = TrackSample(
            statistics.median(item.center for item in neighbours),
            statistics.median(item.rail_z for item in neighbours),
            current.gauge,
            current.score,
        )
    first, last = indexes[0], indexes[-1]
    known = np.array(indexes)
    known_center = np.array([smoothed[i].center for i in indexes])
    known_z = np.array([smoothed[i].rail_z for i in indexes])
    full = np.arange(first, last + 1)
    centers = np.interp(full, known, known_center)
    rail_z = np.interp(full, known, known_z)
    for pos, index in enumerate(full):
        if int(index) not in smoothed:
            smoothed[int(index)] = TrackSample(
                float(centers[pos]), float(rail_z[pos]), config.rail_gauge_m, 0.0
            )
    return smoothed, observed_count


def _fast_track_profile(
    distance: np.ndarray,
    lateral: np.ndarray,
    z: np.ndarray,
    config: DetectorConfig,
    track_model: TrackModel = "greedy",
) -> tuple[dict[int, TrackSample], int]:
    if track_model == "greedy":
        return _greedy_track_profile(distance, lateral, z, config)
    if track_model != "continuity":
        raise ValueError(f"unknown track model: {track_model}")
    continuity = _continuity_track_profile(distance, lateral, z, config)
    if continuity is not None:
        return continuity
    return _greedy_track_profile(distance, lateral, z, config)


def _interpolate_track(
    distance: np.ndarray,
    profile: dict[int, TrackSample],
    config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not profile:
        empty = np.zeros_like(distance, dtype=np.float64)
        return empty, empty, np.zeros_like(distance, dtype=bool)
    indexes = np.array(sorted(profile), dtype=np.int32)
    centers = np.array([profile[int(i)].center for i in indexes])
    rail_z = np.array([profile[int(i)].rail_z for i in indexes])
    position = distance / config.floor_bin_m
    maximum = indexes[-1] + 1 + config.path_extrapolation_m / config.floor_bin_m
    valid = (position >= indexes[0]) & (position <= maximum)
    center = np.interp(position, indexes, centers)
    height = np.interp(position, indexes, rail_z)
    beyond = position > indexes[-1]
    if np.any(beyond) and len(indexes) >= 2:
        tail = min(5, len(indexes))
        index_steps = np.diff(indexes[-tail:])
        center_steps = np.diff(centers[-tail:]) / index_steps
        height_steps = np.diff(rail_z[-tail:]) / index_steps
        center_step = np.clip(
            np.median(center_steps),
            -config.path_center_step_m,
            config.path_center_step_m,
        )
        delta = position[beyond] - indexes[-1]
        center[beyond] = centers[-1] + center_step * delta
        height[beyond] = rail_z[-1] + np.median(height_steps) * delta
    return center, height, valid


def _surface_reference(
    distance: np.ndarray,
    lateral: np.ndarray,
    z: np.ndarray,
    rail_z: np.ndarray,
    config: DetectorConfig,
    half_width_m: float,
) -> np.ndarray:
    d_cell = config.surface_distance_cell_m
    x_cell = config.surface_lateral_cell_m
    di = np.floor(distance / d_cell).astype(np.int32)
    li = np.floor((lateral + half_width_m) / x_cell).astype(np.int32)
    n_d = max(1, int(di.max(initial=0)) + 1)
    n_x = max(1, int(math.ceil(2 * half_width_m / x_cell)) + 1)
    height = z - rail_z
    valid = (
        (li >= 0)
        & (li < n_x)
        & (height >= -0.50)
        & (height <= config.surface_max_height_above_rail_m)
    )
    h_min = -0.50
    h_step = 0.02
    n_h = int(math.ceil((config.surface_max_height_above_rail_m - h_min) / h_step)) + 1
    raw = np.full((n_d, n_x), np.nan, dtype=np.float64)
    if np.any(valid):
        hi = np.clip(np.rint((height[valid] - h_min) / h_step), 0, n_h - 1).astype(
            np.int32
        )
        key = (di[valid] * n_x + li[valid]) * n_h + hi
        histogram = np.bincount(key, minlength=n_d * n_x * n_h).reshape(n_d, n_x, n_h)
        totals = histogram.sum(axis=2)
        threshold = (totals + 1) // 2
        median_index = (histogram.cumsum(axis=2) >= threshold[:, :, None]).argmax(
            axis=2
        )
        raw[totals > 0] = h_min + median_index[totals > 0] * h_step

    radius = config.surface_longitudinal_radius_cells
    smooth = np.full_like(raw, np.nan)
    longitudinal = np.pad(raw, ((radius, radius), (0, 0)), constant_values=np.nan)
    longitudinal_windows = np.lib.stride_tricks.sliding_window_view(
        longitudinal, 2 * radius + 1, axis=0
    )
    neighbours_only = np.concatenate(
        (longitudinal_windows[..., :radius], longitudinal_windows[..., radius + 1 :]),
        axis=-1,
    )
    smooth = _nanmedian_last_axis(neighbours_only)
    padded = np.pad(smooth, 1, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    local = _nanmedian_last_axis(windows.reshape(*windows.shape[:2], -1))
    smooth = np.where(np.isfinite(smooth), smooth, local)
    in_grid = (di >= 0) & (di < n_d) & (li >= 0) & (li < n_x)
    reference = rail_z.copy()
    lookup = np.full(len(distance), np.nan)
    lookup[in_grid] = smooth[di[in_grid], li[in_grid]]
    usable = np.isfinite(lookup)
    reference[usable] = rail_z[usable] + lookup[usable]
    return reference


SupportModel = Literal["frozen", "inverse_square"]


def _minimum_component_points(distance_m: float, model: SupportModel) -> int:
    if model not in ("frozen", "inverse_square"):
        raise ValueError(f"unknown support model: {model}")
    exponent = 0.35 if model == "frozen" else 2.0
    return max(3, round(6 * (30.0 / distance_m) ** exponent))


def _keep_supported_components(
    obstacles: tuple[Obstacle, ...],
    config: DetectorConfig,
    support_model: SupportModel = "frozen",
) -> tuple[Obstacle, ...]:
    """Reject line/single-scan artifacts while retaining a 30 cm box top."""
    kept = []
    for item in obstacles:
        forward_span = item.distance_max_m - item.distance_min_m
        lateral_span = item.lateral_max_m - item.lateral_min_m
        height_span = item.height_max_m - item.height_min_m
        distance = max(config.min_range_m, item.distance_min_m)
        adaptive_points = _minimum_component_points(distance, support_model)
        if item.points < adaptive_points:
            continue
        if height_span < 0.04 and max(forward_span, lateral_span) < 0.10:
            continue
        kept.append(item)
    return tuple(kept)


def detect_fast(
    cloud: PointCloud2,
    config: DetectorConfig = DetectorConfig(),
    context_out: list[tuple[np.ndarray, ...]] | None = None,
    support_model: SupportModel = "frozen",
    track_profile_out: list[dict[int, TrackSample]] | None = None,
    track_model: TrackModel = "greedy",
) -> SafetyDetection:
    arrays = cloud_arrays(cloud)
    x = arrays["x"]
    y = arrays["y"]
    z = arrays["z"]
    intensity = arrays["intensity"]
    ring = arrays["ring"]
    distance = -y
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & ((x != 0) | (y != 0) | (z != 0))
        & (distance >= config.min_range_m)
        & (distance <= config.max_range_m)
        & (np.abs(x) <= config.path_search_half_width_m)
    )
    distance, x, z, intensity, ring = (
        value[valid] for value in (distance, x, z, intensity, ring)
    )
    profile, observed_bins = _fast_track_profile(distance, x, z, config, track_model)
    if track_profile_out is not None:
        track_profile_out.append(profile)
    if not profile:
        return SafetyDetection(
            "UNKNOWN",
            False,
            None,
            0.0,
            0.0,
            0,
            0,
            0,
            None,
            None,
            (),
            "rail_pair_not_observed",
        )
    center, rail_z, supported = _interpolate_track(distance, profile, config)
    lateral = x - center
    component_half_width = config.half_width_m + config.component_guard_band_m
    inside = supported & (np.abs(lateral) <= component_half_width)
    distance, lateral, z, intensity, ring, rail_z = (
        value[inside] for value in (distance, lateral, z, intensity, ring, rail_z)
    )
    surface_z = _surface_reference(
        distance, lateral, z, rail_z, config, component_half_width
    )
    height = z - rail_z
    residual = z - surface_z
    if context_out is not None:
        context_out.append((distance, lateral, height, residual))
    candidate = (
        (height >= -0.05)
        & (height <= config.clearance_height_m)
        & (residual >= config.min_height_above_rail_m)
    )
    if config.roof_taper_start_m is not None:
        fraction = np.clip(
            (height - config.roof_taper_start_m)
            / max(1e-6, config.clearance_height_m - config.roof_taper_start_m),
            0,
            1,
        )
        allowed = (
            config.half_width_m * (1 - fraction)
            + config.roof_half_width_at_top_m * fraction
        )
        candidate &= np.abs(lateral) <= allowed
    points = [
        CandidatePoint(
            float(d),
            float(x),
            float(z_value),
            float(h),
            float(surface_residual),
            float(reflectivity),
            int(ring_id),
        )
        for d, x, z_value, h, surface_residual, reflectivity, ring_id in zip(
            distance[candidate],
            lateral[candidate],
            z[candidate],
            height[candidate],
            residual[candidate],
            intensity[candidate],
            ring[candidate],
        )
    ]
    proposal_config = replace(
        config,
        min_cluster_points=3,
        min_cluster_voxels=1,
        min_cluster_top_height_m=0.04,
        min_cluster_horizontal_extent_m=0.0,
    )
    obstacles = _keep_supported_components(
        _cluster(points, proposal_config), config, support_model
    )
    obstacles = tuple(
        item
        for item in obstacles
        if item.lateral_max_m >= -config.half_width_m
        and item.lateral_min_m <= config.half_width_m
    )
    first, last = min(profile), max(profile)
    reliable_min = first * config.floor_bin_m
    reliable_max = (last + 1) * config.floor_bin_m
    span = max(0.0, reliable_max - reliable_min)
    continuity = observed_bins / max(1, len(profile))
    observability = max(0.0, min(1.0, continuity * min(1.0, span / 60.0)))

    core_obstacles = tuple(
        item
        for item in obstacles
        if abs((item.lateral_min_m + item.lateral_max_m) / 2) <= CORE_HALF_WIDTH_M
    )
    if core_obstacles:
        strongest = max(
            core_obstacles,
            key=lambda item: item.points * max(0.05, item.height_max_m),
        )
        boundary_margin = config.half_width_m - max(
            abs(strongest.lateral_min_m), abs(strongest.lateral_max_m)
        )
        support = min(1.0, strongest.points / 30.0)
        geometry = min(
            1.0,
            max(
                strongest.height_max_m,
                strongest.lateral_max_m - strongest.lateral_min_m,
                strongest.distance_max_m - strongest.distance_min_m,
            )
            / 0.5,
        )
        confidence = max(
            0.05,
            min(
                0.99,
                0.30 * observability
                + 0.30 * support
                + 0.30 * geometry
                + 0.10 * min(1.0, max(0.0, boundary_margin) / 0.3),
            ),
        )
        state = "OBSTACLE"
        reason = "supported_component_inside_core_clearance"
    elif obstacles:
        confidence = max(0.05, min(0.75, 0.25 + 0.5 * observability))
        state = "UNKNOWN"
        reason = "component_only_in_boundary_guard_band"
    elif observability >= 0.30 and span >= 20.0:
        confidence = min(0.99, observability)
        state = "CLEAR"
        reason = "observable_clearance_without_supported_components"
    else:
        confidence = 0.0
        state = "UNKNOWN"
        reason = "insufficient_track_observability"
    return SafetyDetection(
        state=state,
        obstacle=state == "OBSTACLE",
        nearest_distance_m=min(
            (
                o.distance_min_m
                for o in (core_obstacles if core_obstacles else obstacles)
            ),
            default=None,
        ),
        confidence=confidence,
        observability=observability,
        candidate_points=len(points),
        track_bins=len(profile),
        observed_track_bins=observed_bins,
        reliable_range_min_m=reliable_min,
        reliable_range_max_m=reliable_max,
        obstacles=obstacles,
        reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--show-clusters", type=int, default=3)
    args = parser.parse_args()
    processed = 0
    started = time.perf_counter()
    state_counts: dict[str, int] = {}
    for frame, (timestamp, cloud) in enumerate(iter_bag_messages(args.bag)):
        if frame % args.every:
            continue
        frame_started = time.perf_counter()
        result = detect_fast(cloud)
        latency_ms = (time.perf_counter() - frame_started) * 1000
        state_counts[result.state] = state_counts.get(result.state, 0) + 1
        payload = asdict(result)
        payload.update(frame=frame, bag_timestamp_ns=timestamp, latency_ms=latency_ms)
        payload["obstacles"] = [
            asdict(item) for item in result.obstacles[: args.show_clusters]
        ]
        print(json.dumps(payload, ensure_ascii=False))
        processed += 1
        if args.max_frames is not None and processed >= args.max_frames:
            break
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "summary": {
                    "frames": processed,
                    "elapsed_s": elapsed,
                    "fps": processed / elapsed if elapsed else 0.0,
                    "states": state_counts,
                }
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
