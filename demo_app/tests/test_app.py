from __future__ import annotations

from pathlib import Path
import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from demo_app.app import create_app
from demo_app.config import Settings
from demo_app.storage import SQLITE_MAGIC


async def no_worker(_app, _job_id: str, _algorithm: str) -> None:
    return None


async def blocked_worker(_app, _job_id: str, _algorithm: str) -> None:
    await asyncio.Event().wait()


class AppTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        settings = Settings(Path(self.temporary.name), 100, 10, "geometry")
        self.client = TestClient(create_app(settings))

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def test_config_exposes_best_alias(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["best_algorithm"], "geometry")

    def test_presets_can_be_listed_and_loaded(self):
        listing = self.client.get("/api/presets")
        self.assertEqual(listing.status_code, 200)
        self.assertGreaterEqual(len(listing.json()), 3)
        result = self.client.get("/api/presets/obstacle?algorithm=geometry")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["result"]["algorithm"], "geometry")

    def test_preset_rejects_unknown_names(self):
        self.assertEqual(self.client.get("/api/presets/missing").status_code, 404)
        self.assertEqual(
            self.client.get("/api/presets/obstacle?algorithm=missing").status_code,
            422,
        )

    def test_upload_creates_job_with_resolved_algorithm(self):
        with patch("demo_app.app.run_job", no_worker):
            response = self.client.post(
                "/api/jobs",
                files={"bag": ("tiny.db3", SQLITE_MAGIC + b"payload")},
                data={"algorithm": "best"},
            )
        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["algorithm"], "geometry")
        self.assertTrue((Path(self.temporary.name) / payload["job_id"] / "input.db3").is_file())

    def test_upload_rejects_wrong_magic(self):
        response = self.client.post(
            "/api/jobs",
            files={"bag": ("tiny.db3", b"not a database")},
            data={"algorithm": "geometry"},
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_job_when_queue_is_full(self):
        settings = Settings(
            Path(self.temporary.name), 100, 10, "geometry", max_pending_jobs=1
        )
        with TestClient(create_app(settings)) as client:
            with patch("demo_app.app.run_job", blocked_worker):
                first = client.post(
                    "/api/jobs",
                    files={"bag": ("first.db3", SQLITE_MAGIC + b"payload")},
                    data={"algorithm": "geometry"},
                )
                second = client.post(
                    "/api/jobs",
                    files={"bag": ("second.db3", SQLITE_MAGIC + b"payload")},
                    data={"algorithm": "geometry"},
                )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.headers["retry-after"], "30")

    def test_startup_marks_running_job_as_failed(self):
        job_id = "a" * 32
        job_dir = Path(self.temporary.name) / job_id
        job_dir.mkdir()
        (job_dir / "status.json").write_text(
            json.dumps({"status": "running", "job_id": job_id}), encoding="utf-8"
        )
        settings = Settings(Path(self.temporary.name), 100, 10, "geometry")
        with TestClient(create_app(settings)) as client:
            response = client.get(f"/api/jobs/{job_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "failed")
        self.assertIn("restart", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
