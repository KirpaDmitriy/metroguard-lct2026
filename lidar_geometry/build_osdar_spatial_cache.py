from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from lidar_geometry.domain_guard import component_patch
from lidar_geometry.evaluate_osdar_transfer import (
    metro_cloud,
    overlaps_target,
    read_ascii_pcd,
    target_bounds,
)
from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.risk_model import component_features


def build(archives: list[Path], samples_per_class: int, output: Path) -> None:
    patches = []
    components = []
    records = []
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            classes: dict[str, list[str]] = {}
            for name in source.namelist():
                if not name.endswith(".pcd"):
                    continue
                obstacle_class = Path(name).parts[-3]
                classes.setdefault(obstacle_class, []).append(name)
            for obstacle_class, names in sorted(classes.items()):
                positions = np.linspace(0, len(names) - 1, samples_per_class, dtype=int)
                for position in positions:
                    name = sorted(names)[int(position)]
                    with source.open(name) as stream:
                        values, _ = read_ascii_pcd(stream)
                    bounds = target_bounds(values)
                    if bounds is None:
                        continue
                    cloud = metro_cloud(values, pandar_only=False)
                    context = []
                    detection = detect_fast(cloud, context_out=context)
                    matches = [
                        item for item in detection.obstacles
                        if overlaps_target(item, bounds)
                    ]
                    if not matches or not context:
                        continue
                    item = max(matches, key=lambda value: value.points)
                    patches.append(component_patch(context[0], item))
                    components.append(component_features(item))
                    records.append({
                        "archive": archive.name,
                        "class": obstacle_class,
                        "source": name,
                        "points": item.points,
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        patches=np.asarray(patches, dtype=np.float32),
        features=np.asarray(components, dtype=np.float64),
        records_json=json.dumps(records, ensure_ascii=False),
    )
    print(f"{output}: {len(records)} localized OSDaR obstacles")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/artifacts/osdar_spatial_cache.npz"),
    )
    args = parser.parse_args()
    build(args.archives, args.samples_per_class, args.output)


if __name__ == "__main__":
    main()
