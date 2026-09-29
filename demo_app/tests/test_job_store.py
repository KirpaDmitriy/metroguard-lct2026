from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

from demo_app.job_store import cleanup_jobs, read_json, write_json


class JobStoreTest(unittest.TestCase):
    def test_atomic_writes_remain_readable_during_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            write_json(path, {"value": 0})
            failures = []

            def writer():
                for value in range(200):
                    write_json(path, {"value": value})

            thread = threading.Thread(target=writer)
            thread.start()
            while thread.is_alive():
                try:
                    self.assertIsInstance(read_json(path)["value"], int)
                except Exception as error:
                    failures.append(error)
            thread.join()
            self.assertEqual(failures, [])

    def test_cleanup_removes_only_expired_terminal_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            expired = work_dir / ("a" * 32)
            active = work_dir / ("b" * 32)
            fresh = work_dir / ("c" * 32)
            for path, state in (
                (expired, "complete"),
                (active, "running"),
                (fresh, "failed"),
            ):
                path.mkdir()
                write_json(path / "status.json", {"status": state})
            os.utime(expired / "status.json", (10, 10))
            os.utime(active / "status.json", (10, 10))

            removed = cleanup_jobs(work_dir, retention_seconds=50, now=100)

            self.assertEqual(removed, 1)
            self.assertFalse(expired.exists())
            self.assertTrue(active.exists())
            self.assertTrue(fresh.exists())

    def test_cleanup_ignores_non_job_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            unrelated = work_dir / "notes"
            unrelated.mkdir()
            write_json(unrelated / "status.json", {"status": "complete"})
            os.utime(unrelated / "status.json", (10, 10))

            self.assertEqual(cleanup_jobs(work_dir, 50, now=100), 0)
            self.assertTrue(unrelated.exists())
