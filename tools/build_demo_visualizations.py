from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo_app.visualization import compact_cloud
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.pointcloud2 import iter_bag_messages


PRESETS = {
    "obstacle": "for_hackathon/doubleT_obstacle/doubleT_obstacle_0.db3",
    "clean": "for_hackathon/roundT_doubleT/roundT_doubleT_0.db3",
    "synthetic": "synthetic_official/data/cloud_with_fake_obj/cloud_with_fake_obj_0.db3",
}


def result_boxes(preset_dir: Path) -> tuple[dict[int, list[dict]], int]:
    by_frame: dict[int, list[dict]] = {}
    frame_count = 0
    paths = sorted(preset_dir.glob("*_hybrid.json")) + [preset_dir / "geometry.json"]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_count = max(frame_count, len(payload["timeline"]))
        for item in payload["timeline"]:
            by_frame.setdefault(item["frame"], []).extend(item.get("obstacles", ()))
    return by_frame, frame_count


def build(data_root: Path, output_root: Path) -> None:
    for preset_id, relative_bag in PRESETS.items():
        preset_dir = output_root / preset_id
        boxes, frame_count = result_boxes(preset_dir)
        frames = []
        for frame, (_timestamp, cloud) in enumerate(iter_bag_messages(data_root / relative_bag)):
            if frame >= frame_count:
                break
            context = []
            detect_fast(cloud, context_out=context)
            compact = compact_cloud(context[0] if context else None, boxes.get(frame, ()))
            if compact is not None:
                frames.append({"frame": frame, **compact})
        path = preset_dir / "visualization.json"
        path.write_text(
            json.dumps({"frames": frames}, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"{preset_id}: {len(frames)} frames -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("demo_app/presets"))
    args = parser.parse_args()
    build(args.data_root, args.output_root)


if __name__ == "__main__":
    main()
