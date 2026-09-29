from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from demo_app.storage import SQLITE_MAGIC, UploadRejected, save_upload, validate_filename


class MemoryUpload:
    def __init__(self, name: str, payload: bytes):
        self.filename = name
        self.payload = payload
        self.position = 0
        self.closed = False

    async def read(self, size: int) -> bytes:
        chunk = self.payload[self.position:self.position + size]
        self.position += len(chunk)
        return chunk

    async def close(self) -> None:
        self.closed = True


class StorageTest(unittest.TestCase):
    def test_rejects_non_db3_filename(self):
        with self.assertRaises(UploadRejected):
            validate_filename("bag.sqlite")

    def test_streams_sqlite_file(self):
        with tempfile.TemporaryDirectory() as directory:
            upload = MemoryUpload("bag.db3", SQLITE_MAGIC + b"payload")
            destination = Path(directory) / "job" / "input.db3"
            size = asyncio.run(save_upload(upload, destination, 100))
            self.assertEqual(size, len(SQLITE_MAGIC + b"payload"))
            self.assertEqual(destination.read_bytes(), SQLITE_MAGIC + b"payload")
            self.assertTrue(upload.closed)

    def test_rejects_magic_and_size(self):
        for payload, limit in ((b"not sqlite", 100), (SQLITE_MAGIC + b"x" * 20, 16)):
            with self.subTest(payload=payload, limit=limit):
                with tempfile.TemporaryDirectory() as directory:
                    destination = Path(directory) / "job" / "input.db3"
                    with self.assertRaises(UploadRejected):
                        asyncio.run(save_upload(MemoryUpload("bag.db3", payload), destination, limit))
                    self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
