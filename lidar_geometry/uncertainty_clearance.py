from __future__ import annotations

from dataclasses import replace
import math
import statistics

from lidar_geometry.detect_obstacles import DetectorConfig, Obstacle, TrackSample
from lidar_geometry.fast_detector import SafetyDetection


def center_uncertainty_m(
    profile: dict[int, TrackSample],
    distance_m: float,
    config: DetectorConfig = DetectorConfig(),
) -> float:
    target = distance_m / config.floor_bin_m
    observed = sorted(index for index, sample in profile.items() if sample.score > 0)
    local = [index for index in observed if abs(index - target) <= 3]
    if len(local) < 2:
        return 0.20

    x_mean = statistics.mean(local)
    y_mean = statistics.mean(profile[index].center for index in local)
    denominator = sum((index - x_mean) ** 2 for index in local)
    slope = (
        sum(
            (index - x_mean) * (profile[index].center - y_mean)
            for index in local
        )
        / denominator
        if denominator
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    residuals = [
        abs(profile[index].center - (intercept + slope * index))
        for index in local
    ]
    fit_error = 1.4826 * statistics.median(residuals)
    nearest_gap = min(abs(index - target) for index in observed)
    interpolation_error = max(0.0, nearest_gap - 1.0) * config.path_cell_m
    return min(0.30, max(config.path_cell_m / 2, fit_error + interpolation_error))


def confidently_in_clearance(
    component: Obstacle,
    center_uncertainty: float,
    half_width_m: float,
) -> bool:
    if not math.isfinite(center_uncertainty) or center_uncertainty < 0:
        raise ValueError("center uncertainty must be finite and non-negative")
    return (
        component.lateral_min_m <= half_width_m - center_uncertainty
        and component.lateral_max_m >= -half_width_m + center_uncertainty
    )


def apply_uncertainty_gate(
    detection: SafetyDetection,
    profile: dict[int, TrackSample],
    config: DetectorConfig = DetectorConfig(),
) -> SafetyDetection:
    if detection.state != "OBSTACLE" or not detection.obstacles:
        return detection
    confident = []
    for component in detection.obstacles:
        distance = (component.distance_min_m + component.distance_max_m) / 2
        uncertainty = center_uncertainty_m(profile, distance, config)
        if confidently_in_clearance(component, uncertainty, config.half_width_m):
            confident.append(component)
    if confident:
        return detection
    return replace(
        detection,
        state="UNKNOWN",
        obstacle=False,
        confidence=min(detection.confidence, 0.5),
        reason="clearance_membership_within_rail_fit_uncertainty",
    )
