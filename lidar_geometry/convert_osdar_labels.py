"""Flatten OSDaR23 OpenLABEL lidar annotations into JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(source: Path, output: Path) -> int:
    document = json.loads(source.read_text(encoding="utf-8"))["openlabel"]
    definitions = document.get("objects", {})
    records = []
    for frame_id, frame in document.get("frames", {}).items():
        lidar_uri = frame["frame_properties"]["streams"]["lidar"]["uri"]
        for object_id, instance in frame.get("objects", {}).items():
            definition = definitions[object_id]
            object_data = instance.get("object_data", {})
            cuboids = [
                item for item in object_data.get("cuboid", [])
                if item.get("coordinate_system") == "lidar"
            ]
            point_vectors = [
                item for item in object_data.get("vec", [])
                if item.get("coordinate_system") == "lidar"
            ]
            if not cuboids and not point_vectors:
                continue
            records.append({
                "frame": int(frame_id),
                "lidar_uri": lidar_uri,
                "object_id": object_id,
                "type": definition.get("type"),
                "cuboid_xyz_quaternion_size": cuboids[0]["val"] if cuboids else None,
                "point_indices": point_vectors[0]["val"] if point_vectors else [],
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps({"records": convert(args.source, args.output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
