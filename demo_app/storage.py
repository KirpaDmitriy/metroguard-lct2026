from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile


SQLITE_MAGIC = b"SQLite format 3\x00"
CHUNK_BYTES = 1024 * 1024


class UploadRejected(ValueError):
    pass


def validate_filename(filename: str | None) -> None:
    if not filename or Path(filename).suffix.lower() != ".db3":
        raise UploadRejected("Only .db3 rosbag files are accepted")


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    validate_filename(upload.filename)
    destination.parent.mkdir(parents=True, exist_ok=False)
    size = 0
    prefix = bytearray()
    try:
        with destination.open("xb") as output:
            while chunk := await upload.read(CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadRejected(
                        f"File exceeds the {max_bytes // 1024**2} MiB upload limit"
                    )
                if len(prefix) < len(SQLITE_MAGIC):
                    needed = len(SQLITE_MAGIC) - len(prefix)
                    prefix.extend(chunk[:needed])
                output.write(chunk)
        if bytes(prefix) != SQLITE_MAGIC:
            raise UploadRejected("The file is not an SQLite 3 database")
        return size
    except BaseException:
        destination.unlink(missing_ok=True)
        destination.parent.rmdir()
        raise
    finally:
        await upload.close()
