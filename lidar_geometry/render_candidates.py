"""Render geometric candidates for manual component-level review."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from lidar_geometry.detect_obstacles import DetectorConfig, _candidate_points
from lidar_geometry.pointcloud2 import iter_bag_messages


def circles(points, xmap, ymap, color: str, radius: float) -> str:
    return "".join(
        f'<circle cx="{xmap(point):.1f}" cy="{ymap(point):.1f}" r="{radius}" fill="{color}"/>'
        for point in points
    )


def render_ppm(candidate: dict, points, output: Path) -> None:
    """Dependency-free raster: left is top view, right is side view."""
    d0, d1 = candidate["distance_min_m"], candidate["distance_max_m"]
    center = (d0 + d1) / 2
    context_min, context_max = max(0.0, center - 6), center + 6
    nearby = [point for point in points if context_min <= point.distance <= context_max]
    width, height = 1400, 700
    pixels = bytearray((15, 21, 28)) * (width * height)

    def dot(x: float, y: float, color: tuple[int, int, int], radius: int) -> None:
        cx, cy = round(x), round(y)
        for py in range(max(0, cy - radius), min(height, cy + radius + 1)):
            for px in range(max(0, cx - radius), min(width, cx + radius + 1)):
                offset = (py * width + px) * 3
                pixels[offset : offset + 3] = bytes(color)

    def selected(point) -> bool:
        return (
            d0 - 0.08 <= point.distance <= d1 + 0.08
            and candidate["lateral_min_m"] - 0.08 <= point.lateral <= candidate["lateral_max_m"] + 0.08
            and candidate["height_min_m"] - 0.08 <= point.height <= candidate["height_max_m"] + 0.08
        )

    for point in nearby:
        color, radius = ((255, 59, 48), 3) if selected(point) else ((100, 125, 145), 1)
        top_x = 25 + (point.lateral + 1.05) / 2.1 * 650
        top_y = 25 + (point.distance - context_min) / (context_max - context_min) * 650
        side_x = 725 + (point.distance - context_min) / (context_max - context_min) * 650
        side_y = 25 + (3.05 - point.height) / 3.15 * 650
        dot(top_x, top_y, color, radius)
        dot(side_x, side_y, color, radius)
    output.write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)


def render(candidate: dict, points, output: Path, candidate_id: str) -> None:
    d0, d1 = candidate["distance_min_m"], candidate["distance_max_m"]
    center = (d0 + d1) / 2
    context_min, context_max = max(0.0, center - 6), center + 6
    nearby = [point for point in points if context_min <= point.distance <= context_max]
    selected = [
        point for point in nearby
        if d0 - 0.08 <= point.distance <= d1 + 0.08
        and candidate["lateral_min_m"] - 0.08 <= point.lateral <= candidate["lateral_max_m"] + 0.08
        and candidate["height_min_m"] - 0.08 <= point.height <= candidate["height_max_m"] + 0.08
    ]
    width, height = 1400, 760
    left_x0, plot_y0, plot_w, plot_h = 70, 100, 590, 590
    right_x0 = 750

    top_x = lambda p: left_x0 + (p.lateral + 1.05) / 2.1 * plot_w
    top_y = lambda p: plot_y0 + (p.distance - context_min) / (context_max - context_min) * plot_h
    side_x = lambda p: right_x0 + (p.distance - context_min) / (context_max - context_min) * plot_w
    side_y = lambda p: plot_y0 + (3.05 - p.height) / 3.15 * plot_h
    title = (
        f"{candidate_id} | frame={candidate['frame']} | d={d0:.1f}..{d1:.1f} m | "
        f"supervised={candidate['supervised_score']:.3f} | anomaly={candidate['anomaly_score']:.1f}"
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#10151c"/>',
        f'<text x="40" y="45" fill="white" font-size="22" font-family="monospace">{html.escape(title)}</text>',
        f'<rect x="{left_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="#18222d" stroke="#607080"/>',
        f'<rect x="{right_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" fill="#18222d" stroke="#607080"/>',
        f'<text x="{left_x0}" y="85" fill="white" font-size="18">TOP: lateral / distance</text>',
        f'<text x="{right_x0}" y="85" fill="white" font-size="18">SIDE: distance / height</text>',
        circles(nearby, top_x, top_y, "#667788", 1.4),
        circles(nearby, side_x, side_y, "#667788", 1.4),
        circles(selected, top_x, top_y, "#ff3b30", 3.0),
        circles(selected, side_x, side_y, "#ff3b30", 3.0),
        f'<text x="70" y="730" fill="#c8d2dc" font-size="16">red=candidate; gray=other residual components; top forward direction is down</text>',
        "</svg>",
    ]
    output.write_text("".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("lidar_geometry/artifacts/review"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    candidates = report["top_obstacle_bag_candidates"][: args.limit]
    by_frame: dict[int, list[tuple[int, dict]]] = {}
    for index, item in enumerate(candidates):
        by_frame.setdefault(item["frame"], []).append((index, item))
    args.output.mkdir(parents=True, exist_ok=True)
    labels = []
    for frame, (_timestamp, cloud) in enumerate(iter_bag_messages(args.bag)):
        if frame not in by_frame:
            continue
        points, _, _ = _candidate_points(cloud, DetectorConfig())
        for index, candidate in by_frame[frame]:
            candidate_id = f"f{frame:04d}_c{index:02d}"
            render(candidate, points, args.output / f"{candidate_id}.svg", candidate_id)
            render_ppm(candidate, points, args.output / f"{candidate_id}.ppm")
            labels.append({
                "id": candidate_id,
                "frame": frame,
                "distance_min_m": candidate["distance_min_m"],
                "label": "unreviewed",
                "allowed_labels": ["obstacle", "person", "infrastructure", "noise", "uncertain"],
                "comment": "",
            })
    (args.output / "labels.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in labels) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rendered": len(labels), "output": str(args.output)}))


if __name__ == "__main__":
    main()
