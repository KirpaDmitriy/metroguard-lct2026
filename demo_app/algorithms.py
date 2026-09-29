from __future__ import annotations

from pathlib import Path

from lidar_geometry.fast_detector import detect_fast
from lidar_geometry.hybrid_detector import filter_geometric_detection
from lidar_geometry.runtime_detector import ALGORITHMS, RuntimeDetector
from lidar_geometry.runtime_detector import filter_portable_tree_detection


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
        elif self.tree is not None:
            detection = filter_portable_tree_detection(geometric, self.tree)
        else:
            detection = geometric
        return detection, context[0] if context else None
