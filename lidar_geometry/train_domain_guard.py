from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lidar_geometry.domain_guard import DomainGuard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("lidar_geometry/artifacts/spatial_patch_cache_v1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lidar_geometry/models/domain_guard.npz"),
    )
    args = parser.parse_args()
    data = np.load(args.cache, allow_pickle=False)
    records = json.loads(str(data["records_json"]))
    real = np.asarray([record["source"] == "real" for record in records])
    guard = DomainGuard.fit(data["patches"][real])
    guard.save(args.output)
    print(f"{args.output}: {len(guard.references)} references, radius={guard.radius:.6f}")


if __name__ == "__main__":
    main()
