from __future__ import annotations

from pathlib import Path

from lidar_geometry.runtime_detector import ALGORITHMS, RuntimeDetector


class Detector(RuntimeDetector):
    def __init__(self, name: str, root: Path):
        self.name = name
        super().__init__(
            name,
            risk_model_path=root / "lidar_geometry/models/risk_model.json",
            portable_model_path=(
                root / "lidar_geometry/artifacts/linear_extra_trees_hybrid.npz"
            ),
        )
