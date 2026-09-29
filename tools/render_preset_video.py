import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1600
HEIGHT = 900
FPS = 10


def load_font(size: int, bold: bool = False):
    candidates = (
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def contains(point: list[float], obstacle: dict) -> bool:
    lateral, distance, height = point[:3]
    return (
        obstacle["distance_min_m"] - 0.08
        <= distance
        <= obstacle["distance_max_m"] + 0.08
        and obstacle["lateral_min_m"] - 0.08
        <= lateral
        <= obstacle["lateral_max_m"] + 0.08
        and obstacle["height_min_m"] - 0.08 <= height <= obstacle["height_max_m"] + 0.08
    )


def draw_frame(item: dict, points: list[list[float]], title: str) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#07111a")
    draw = ImageDraw.Draw(image, "RGBA")
    obstacles = item.get("obstacles", [])
    range_m = min(
        200,
        max(60, ((max((point[1] for point in points), default=60) + 19) // 20) * 20),
    )
    panels = ((40, 130, 510, 800), (565, 130, 1035, 800), (1090, 130, 1560, 800))
    labels = ("ВИД СВЕРХУ", "ВИД СБОКУ", "ПЕРСПЕКТИВА")
    for panel, label in zip(panels, labels):
        draw.rounded_rectangle(panel, 18, fill="#0c1823", outline="#26394a", width=2)
        draw.text((panel[0], 88), label, fill="#dbe5ee", font=load_font(23, True))
    draw.text((40, 28), title, fill="#f4f7fb", font=load_font(34, True))
    state = item["state"]
    state_color = {"OBSTACLE": "#ff6577", "CLEAR": "#62d6aa"}.get(state, "#ffc857")
    state_label = {"OBSTACLE": "ПРЕПЯТСТВИЕ", "CLEAR": "ПУТЬ СВОБОДЕН"}.get(
        state, "НЕДОСТАТОЧНО ДАННЫХ"
    )
    draw.text(
        (WIDTH - 430, 35), state_label, fill=state_color, font=load_font(27, True)
    )

    top, side, perspective = panels
    top_x = lambda lateral: top[0] + (lateral + 2.4) / 4.8 * (top[2] - top[0])
    top_y = lambda distance: top[3] - distance / range_m * (top[3] - top[1])
    side_x = lambda distance: side[0] + distance / range_m * (side[2] - side[0])
    side_y = lambda height: side[3] - (height + 0.25) / 3.5 * (side[3] - side[1])

    draw.rectangle(
        (top_x(-1.05), top[1], top_x(1.05), top[3]),
        fill="#62d6aa12",
        outline="#62d6aa",
        width=2,
    )
    draw.rectangle(
        (side[0], side_y(3), side[2], side_y(0)),
        fill="#62d6aa12",
        outline="#62d6aa",
        width=2,
    )
    horizon_y = perspective[1] + 45
    center_x = (perspective[0] + perspective[2]) / 2
    draw.line(
        (perspective[0] + 25, perspective[3], center_x - 12, horizon_y),
        fill="#62d6aa",
        width=2,
    )
    draw.line(
        (perspective[2] - 25, perspective[3], center_x + 12, horizon_y),
        fill="#62d6aa",
        width=2,
    )

    for point in points:
        hit = any(contains(point, obstacle) for obstacle in obstacles)
        color = "#ff6577" if hit else "#75a9cf9a"
        radius = 3 if hit else 1
        draw.ellipse(
            (
                top_x(point[0]) - radius,
                top_y(point[1]) - radius,
                top_x(point[0]) + radius,
                top_y(point[1]) + radius,
            ),
            fill=color,
        )
        draw.ellipse(
            (
                side_x(point[1]) - radius,
                side_y(point[2]) - radius,
                side_x(point[1]) + radius,
                side_y(point[2]) + radius,
            ),
            fill=color,
        )
        depth = min(1, max(0, point[1] / range_m))
        scale = 1 - 0.82 * depth
        px = center_x + point[0] * 92 * scale
        py = (
            perspective[3]
            - 70
            - depth * (perspective[3] - horizon_y - 70)
            - point[2] * 72 * scale
        )
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    draw.text(
        (40, 830),
        f"Кадр {item['frame']} · диапазон 0–{int(range_m)} м · голубые точки — облако · красные — подтверждённая компонента",
        fill="#91a2b3",
        font=load_font(20),
    )
    return image


def render(
    result_path: Path, cloud_path: Path, output: Path, title: str, frames: int
) -> None:
    timeline = json.loads(result_path.read_text(encoding="utf-8"))["timeline"]
    cloud_frames = json.loads(cloud_path.read_text(encoding="utf-8"))["frames"]
    by_frame = {item["frame"]: item["points"] for item in cloud_frames}
    available = [item for item in timeline if item["frame"] in by_frame]
    obstacle_index = next(
        (index for index, item in enumerate(available) if item["state"] == "OBSTACLE"),
        None,
    )
    start = max(0, obstacle_index - frames // 3) if obstacle_index is not None else 0
    selected = available[start : start + frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for item in selected:
        process.stdin.write(draw_frame(item, by_frame[item["frame"]], title).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    render(args.result, args.cloud, args.output, args.title, args.frames)


if __name__ == "__main__":
    main()
