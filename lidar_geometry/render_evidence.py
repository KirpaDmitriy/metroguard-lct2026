"""Render a reproducible evidence card from exported demo data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path,
                        default=Path("lidar_geometry/demo/demo-data.json"), nargs="?")
    parser.add_argument("--output", type=Path,
                        default=Path("lidar_geometry/artifacts/metroguard_evidence.png"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    scene = data["scenes"][0]
    image = Image.new("RGB", (1600, 900), "#07100f")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, 1600, 105), fill="#10201d")
    draw.text((52, 27), "METROGUARD / REAL ROS 2 FRAME", fill="#69f5bd",
              font=font(25, True))
    draw.text((52, 60), "doubleT_obstacle · frame 10 · x / −y / z",
              fill="#82938f", font=font(20))
    panels = [(42, 140, 1010, 850), (1040, 140, 1558, 850)]
    for box in panels:
        draw.rounded_rectangle(box, radius=24, fill="#0d1917", outline="#1e3631", width=2)
    plot = (78, 195, 974, 810)
    for i in range(12):
        y = plot[3] - i / 11 * (plot[3] - plot[1])
        draw.line((plot[0], y, plot[2], y), fill="#17302b", width=1)
    def sx(x): return (plot[0] + plot[2]) / 2 + x / 8 * (plot[2] - plot[0]) / 2
    def sy(y): return plot[3] - y / 110 * (plot[3] - plot[1])
    for x, y, z in scene["points"]:
        if 0 <= y <= 110 and abs(x) <= 8:
            shade = max(70, min(210, int(100 + z * 20)))
            draw.point((sx(x), sy(y)), fill=(90, shade, 155, 115))
    for edge in (-1.05, 1.05):
        draw.line((sx(edge), sy(0), sx(edge), sy(110)), fill="#69f5bd", width=3)
    for item in scene["components"]:
        draw.rectangle((sx(item["lateral_min_m"]), sy(item["distance_max_m"]),
                        sx(item["lateral_max_m"]), sy(item["distance_min_m"])),
                       outline="#ff6d69", width=3)
    draw.text((78, 155), "ВИД СВЕРХ · 110 М", fill="#eef4f2", font=font(20, True))
    q = scene["decision"]
    draw.text((1080, 190), q["state"], fill="#ffcf67", font=font(54, True))
    draw.text((1080, 270), "БЛИЖАЙШИЙ КАНДИДАТ", fill="#82938f", font=font(17, True))
    draw.text((1080, 305), f'{q["distance_m"]:.1f} м', fill="#eef4f2", font=font(55, True))
    draw.text((1080, 405), "УВЕРЕННОСТЬ", fill="#82938f", font=font(17, True))
    draw.text((1080, 440), f'{q["confidence"]:.0%}', fill="#eef4f2", font=font(42, True))
    draw.text((1080, 525), "НАБЛЮДАЕМОСТЬ", fill="#82938f", font=font(17, True))
    draw.text((1080, 560), f'{q["observability"]:.0%}', fill="#eef4f2", font=font(42, True))
    draw.rounded_rectangle((1080, 660, 1515, 780), radius=16,
                           fill="#132a25", outline="#285346", width=2)
    draw.text((1105, 685), "geometry + risk ranker\n+ 2-of-3 temporal gate",
              fill="#69f5bd", font=font(21, True), spacing=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
