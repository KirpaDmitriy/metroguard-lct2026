from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lidar_geometry.detect_obstacles import DetectorConfig
from lidar_geometry.fast_detector import (
    _fast_track_profile,
    _interpolate_track,
    cloud_arrays,
)
from lidar_geometry.pointcloud2 import iter_bag_messages
from lidar_geometry.scenario_catalog import SCENARIOS as SCENARIO_CATALOG


SCENARIOS = (
    "2x2 center",
    "0.3x0.3 center",
    "0.3x0.3 on rail",
    "0.3x0.3 edge inside",
    "0.3x0.3 outside",
    "2x2 edge inside",
    "2x2 outside",
    "2x2 above",
    "2x0.2 on rails",
    "0.05 hanging",
)


def font(size: int, bold: bool = False):
    suffix = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / suffix
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def render_tile(cloud, frame: int, title: str, scenario, focus_distance_m: float) -> Image.Image:
    image = Image.new("RGB", (900, 380), "#091311")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((18, 12), f"{title} · frame {frame}", fill="#ecf5f1", font=font(18, True))
    arrays = cloud_arrays(cloud)
    raw_lateral = arrays["x"]
    distance = -arrays["y"]
    z = arrays["z"]
    config = DetectorConfig()
    path_mask = (
        np.isfinite(distance) & np.isfinite(raw_lateral) & np.isfinite(z)
        & (distance >= config.min_range_m) & (distance <= config.max_range_m)
        & (np.abs(raw_lateral) <= config.path_search_half_width_m)
    )
    profile, _ = _fast_track_profile(
        distance[path_mask], raw_lateral[path_mask], z[path_mask], config
    )
    center, rail_z, supported = _interpolate_track(distance, profile, config)
    lateral = raw_lateral - center
    height = z - rail_z
    distance_min = focus_distance_m - 12
    distance_max = focus_distance_m + 12
    valid = (
        np.isfinite(lateral) & np.isfinite(distance) & np.isfinite(z)
        & supported & (distance >= distance_min) & (distance <= distance_max)
        & (np.abs(lateral) <= 3) & (height >= -0.5) & (height <= 5.5)
    )
    lateral, distance, height = (
        lateral[valid][::2], distance[valid][::2], height[valid][::2]
    )
    panels = ((24, 55, 425, 345), (475, 55, 876, 345))
    for box in panels:
        draw.rectangle(box, fill="#101e1b", outline="#36534c", width=1)
    draw.text((28, 57), "TOP", fill="#9bb0aa", font=font(13, True))
    draw.text((479, 57), "SIDE", fill="#9bb0aa", font=font(13, True))

    def top_x(value):
        return 24 + (value + 3) / 6 * 401

    def distance_y(value):
        return 345 - (value - distance_min) / (distance_max - distance_min) * 290

    def side_x(value):
        return 475 + (value - distance_min) / (distance_max - distance_min) * 401

    def side_y(value):
        return 345 - (value + 0.5) / 6 * 290

    for edge in (-1.05, 1.05):
        draw.line((
            top_x(edge), distance_y(distance_min),
            top_x(edge), distance_y(distance_max),
        ), fill="#62e7ad", width=2)
    for clearance_height in (0.0, 3.0):
        draw.line((
            side_x(distance_min), side_y(clearance_height),
            side_x(distance_max), side_y(clearance_height),
        ), fill="#62e7ad", width=2)

    inside = (np.abs(lateral) <= 1.05) & (height >= 0.0) & (height <= 3.0)
    for x, d, point_height, is_inside in zip(lateral, distance, height, inside):
        color = (255, 102, 99, 135) if is_inside else (112, 145, 156, 80)
        draw.point((top_x(float(x)), distance_y(float(d))), fill=color)
        draw.point((side_x(float(d)), side_y(float(point_height))), fill=color)
    target = scenario.obstacle
    d0 = focus_distance_m - target.length_m / 2
    d1 = focus_distance_m + target.length_m / 2
    x0 = target.lateral_m - target.width_m / 2
    x1 = target.lateral_m + target.width_m / 2
    h0 = target.bottom_m
    h1 = target.bottom_m + target.height_m
    draw.rectangle(
        (top_x(x0), distance_y(d1), top_x(x1), distance_y(d0)),
        outline="#ffd166", width=2,
    )
    draw.rectangle(
        (side_x(d0), side_y(h1), side_x(d1), side_y(h0)),
        outline="#ffd166", width=2,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, default=Path("lidar_geometry/artifacts/official_scenario_review.png"))
    parser.add_argument("--frames", nargs=10, type=int, default=[150 * index for index in range(1, 11)])
    parser.add_argument("--focus-distance", type=float, default=40.0)
    args = parser.parse_args()
    wanted = dict(zip(args.frames, zip(SCENARIOS, SCENARIO_CATALOG[:10])))
    tiles = []
    for frame, (_, cloud) in enumerate(iter_bag_messages(args.bag)):
        if frame in wanted:
            title, scenario = wanted[frame]
            tiles.append((frame, render_tile(
                cloud, frame, title, scenario, args.focus_distance
            )))
        if frame > max(wanted):
            break
    sheet = Image.new("RGB", (1800, 1900), "#07100f")
    for index, (_, tile) in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 900, (index // 2) * 380))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"{args.output} ({len(tiles)} scenarios)")


if __name__ == "__main__":
    main()
