from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time


ACTIVE_STATES = {"queued", "running"}
TERMINAL_STATES = {"complete", "failed"}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def reconcile_jobs(work_dir: Path) -> int:
    reconciled = 0
    for job_dir in _job_directories(work_dir):
        status_path = job_dir / "status.json"
        try:
            status = read_json(status_path)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            continue
        if status.get("status") not in ACTIVE_STATES:
            continue
        if (job_dir / "result.json").is_file():
            status.update(
                status="complete",
                result_url=f"/api/jobs/{job_dir.name}/result",
            )
        else:
            status.update(status="failed", error="Interrupted by server restart")
        write_json(status_path, status)
        reconciled += 1
    return reconciled


def cleanup_jobs(work_dir: Path, retention_seconds: int, now: float | None = None) -> int:
    if retention_seconds <= 0:
        return 0
    cutoff = (time.time() if now is None else now) - retention_seconds
    removed = 0
    for job_dir in _job_directories(work_dir):
        status_path = job_dir / "status.json"
        try:
            status = read_json(status_path)
            modified = status_path.stat().st_mtime
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            continue
        if status.get("status") in TERMINAL_STATES and modified < cutoff:
            shutil.rmtree(job_dir)
            removed += 1
    return removed


def _job_directories(work_dir: Path):
    if not work_dir.is_dir():
        return
    for path in work_dir.iterdir():
        if (
            path.is_dir()
            and not path.is_symlink()
            and len(path.name) == 32
            and all(character in "0123456789abcdef" for character in path.name)
        ):
            yield path
