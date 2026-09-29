from __future__ import annotations

from pathlib import Path

from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
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

    def analyze(self, cloud):
        context = []
        geometric = detect_fast(cloud, self.config, context_out=context)
        if self.linear is not None:
            detection = filter_geometric_detection(geometric, self.linear)
        else:
            detection = self.apply_geometric(geometric)
        return detection, context[0] if context else None
