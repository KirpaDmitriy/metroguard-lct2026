from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("algorithms.json")


@dataclass(frozen=True)
class Settings:
    work_dir: Path
    max_upload_bytes: int
    job_timeout_seconds: int
    best_algorithm: str
    max_pending_jobs: int = 4
    job_retention_seconds: int = 24 * 3600
    cleanup_interval_seconds: int = 3600


def load_settings() -> Settings:
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    best = os.getenv("METROGUARD_BEST_ALGORITHM", payload["best_algorithm"])
    if best not in {"geometry", "linear_hybrid", "tree_hybrid", "memory_hybrid"}:
        raise ValueError(f"Unsupported best algorithm: {best}")
    settings = Settings(
        work_dir=Path(os.getenv("METROGUARD_WORK_DIR", "/tmp/metroguard-demo")),
        max_upload_bytes=int(os.getenv("METROGUARD_MAX_UPLOAD_BYTES", 8 * 1024**3)),
        job_timeout_seconds=int(os.getenv("METROGUARD_JOB_TIMEOUT_SECONDS", 3600)),
        best_algorithm=best,
        max_pending_jobs=int(os.getenv("METROGUARD_MAX_PENDING_JOBS", 4)),
        job_retention_seconds=int(
            os.getenv("METROGUARD_JOB_RETENTION_SECONDS", 24 * 3600)
        ),
        cleanup_interval_seconds=int(
            os.getenv("METROGUARD_CLEANUP_INTERVAL_SECONDS", 3600)
        ),
    )
    if settings.max_pending_jobs < 1:
        raise ValueError("METROGUARD_MAX_PENDING_JOBS must be positive")
    if settings.cleanup_interval_seconds < 1:
        raise ValueError("METROGUARD_CLEANUP_INTERVAL_SECONDS must be positive")
    return settings
