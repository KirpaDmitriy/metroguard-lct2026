from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False):
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def contains(point: list[float], obstacle: dict) -> bool:
    lateral, distance, height = point[:3]
    return (
        obstacle["distance_min_m"] - 0.08 <= distance <= obstacle["distance_max_m"] + 0.08
        and obstacle["lateral_min_m"] - 0.08 <= lateral <= obstacle["lateral_max_m"] + 0.08
        and obstacle["height_min_m"] - 0.08 <= height <= obstacle["height_max_m"] + 0.08
    )


def render(result_path: Path, cloud_path: Path, output: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    clouds = json.loads(cloud_path.read_text(encoding="utf-8"))["frames"]
    cloud_by_frame = {item["frame"]: item["points"] for item in clouds}
    item = next(
        value for value in result["timeline"]
        if value["state"] == "OBSTACLE" and value["frame"] in cloud_by_frame
    )
    points = cloud_by_frame[item["frame"]]
    obstacles = item["obstacles"]
    nearest = min(obstacles, key=lambda value: value["distance_min_m"])
    distance_limit = min(
        150,
        max(60, math.ceil((nearest["distance_max_m"] + 20) / 20) * 20),
    )

    image = Image.new("RGB", (1500, 660), "#07111a")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((28, 82, 950, 625), 22, fill="#0c1823", outline="#26394a", width=2)
    draw.rounded_rectangle((980, 82, 1472, 625), 22, fill="#0c1823", outline="#26394a", width=2)
    draw.text((42, 24), "ЧТО УВИДЕЛ METROGUARD", fill="#6ee7b7", font=font(28, True))
    draw.text((1000, 116), "ПРЕПЯТСТВИЕ", fill="#ff6b7a", font=font(38, True))
    draw.text((1000, 180), f'{item["distance_m"]:.1f} м', fill="#f4f7fb", font=font(64, True))
    draw.text((1000, 265), "АНСАМБЛЬ ИЗ 100 ДЕРЕВЬЕВ", fill="#8fa2b5", font=font(17, True))
    draw.text((1000, 300), f'уверенность {item["confidence"]:.0%}', fill="#dbe5ee", font=font(25, True))
    draw.text((1000, 365), "Красным выделены точки,\nкоторые сформировали\nподтверждённую компоненту.", fill="#b4c1cd", font=font(23), spacing=10)

    top = (60, 135, 485, 580)
    side = (515, 135, 925, 580)
    draw.text((60, 100), "ВИД СВЕРХУ", fill="#dbe5ee", font=font(18, True))
    draw.text((515, 100), "ВИД СБОКУ", fill="#dbe5ee", font=font(18, True))

    def top_x(lateral):
        return top[0] + (lateral + 2.4) / 4.8 * (top[2] - top[0])

    def top_y(distance):
        return top[3] - distance / distance_limit * (top[3] - top[1])

    def side_x(distance):
        return side[0] + distance / distance_limit * (side[2] - side[0])

    def side_y(height):
        return side[3] - (height + 0.25) / 3.5 * (side[3] - side[1])

    draw.rectangle((top_x(-1.05), top[1], top_x(1.05), top[3]), fill="#62d6aa14", outline="#62d6aa", width=2)
    draw.rectangle((side[0], side_y(3.0), side[2], side_y(0)), fill="#62d6aa14", outline="#62d6aa", width=2)
    for point in points:
        if point[1] > distance_limit:
            continue
        hit = any(contains(point, obstacle) for obstacle in obstacles)
        color = "#ff6577" if hit else "#75a9cf99"
        radius = 3 if hit else 1
        draw.ellipse((top_x(point[0]) - radius, top_y(point[1]) - radius,
                      top_x(point[0]) + radius, top_y(point[1]) + radius), fill=color)
        draw.ellipse((side_x(point[1]) - radius, side_y(point[2]) - radius,
                      side_x(point[1]) + radius, side_y(point[2]) + radius), fill=color)
    draw.rectangle((top_x(nearest["lateral_min_m"]), top_y(nearest["distance_max_m"]),
                    top_x(nearest["lateral_max_m"]), top_y(nearest["distance_min_m"])),
                   outline="#ff6577", width=4)
    draw.rectangle((side_x(nearest["distance_min_m"]), side_y(nearest["height_max_m"]),
                    side_x(nearest["distance_max_m"]), side_y(nearest["height_min_m"])),
                   outline="#ff6577", width=4)
    draw.text((60, 598), f"0–{distance_limit} м · официальная синтетика организаторов · кадр {item['frame']}", fill="#8092a3", font=font(16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.result, args.cloud, args.output)


if __name__ == "__main__":
    main()
