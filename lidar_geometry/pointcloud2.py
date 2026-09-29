"""Minimal ROS 2 PointCloud2 CDR reader with no ROS/Python dependencies.

The hackathon bags store uncompressed ``sensor_msgs/msg/PointCloud2`` messages
inside an SQLite3 rosbag2 database.  This module intentionally uses only the
Python standard library so it can also be used for quick offline experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sqlite3
import struct
from pathlib import Path
from typing import Iterator


_POINT_FIELD_FORMATS = {
    1: "b",   # INT8
    2: "B",   # UINT8
    3: "h",   # INT16
    4: "H",   # UINT16
    5: "i",   # INT32
    6: "I",   # UINT32
    7: "f",   # FLOAT32
    8: "d",   # FLOAT64
}


class CdrReader:
    def __init__(self, payload: bytes):
        if len(payload) < 4:
            raise ValueError("CDR payload is too short")
        # ROS 2 CDR encapsulation 0x0001 is little-endian, 0x0000 big-endian.
        encapsulation = int.from_bytes(payload[:2], "big")
        self.endian = "<" if encapsulation == 1 else ">"
        self.payload = payload
        self.pos = 4

    def align(self, size: int) -> None:
        self.pos += (-self.pos) % size

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        self.align(size)
        value = struct.unpack_from(self.endian + fmt, self.payload, self.pos)[0]
        self.pos += size
        return value

    def u8(self) -> int:
        return self.unpack("B")

    def u32(self) -> int:
        return self.unpack("I")

    def i32(self) -> int:
        return self.unpack("i")

    def string(self) -> str:
        size = self.u32()
        raw = self.payload[self.pos : self.pos + size]
        self.pos += size
        return raw.rstrip(b"\0").decode("utf-8")


@dataclass(frozen=True)
class PointField:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class PointCloud2:
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    height: int
    width: int
    fields: tuple[PointField, ...]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: memoryview
    is_dense: bool

    @property
    def point_count(self) -> int:
        return self.height * self.width

    def _field_unpacker(self, names: tuple[str, ...]) -> struct.Struct:
        by_name = {field.name: field for field in self.fields}
        missing = set(names) - set(by_name)
        if missing:
            raise ValueError(f"PointCloud2 fields missing: {sorted(missing)}")

        ordered = sorted((by_name[name] for name in names), key=lambda field: field.offset)
        fmt = ">" if self.is_bigendian else "<"
        cursor = 0
        for field in ordered:
            if field.count != 1:
                raise ValueError(f"Array field {field.name!r} is not supported")
            code = _POINT_FIELD_FORMATS.get(field.datatype)
            if code is None:
                raise ValueError(f"Unsupported PointField datatype {field.datatype}")
            fmt += f"{field.offset - cursor}x{code}"
            cursor = field.offset + struct.calcsize(code)
        fmt += f"{self.point_step - cursor}x"
        return struct.Struct(fmt)

    def iter_xyzirt(self) -> Iterator[tuple[float, float, float, float, int, float]]:
        """Yield x, y, z, intensity, ring, timestamp in physical field order."""
        names = ("x", "y", "z", "intensity", "ring", "timestamp")
        by_name = {field.name: field for field in self.fields}
        physical_names = tuple(
            field.name for field in sorted((by_name[name] for name in names), key=lambda f: f.offset)
        )
        unpacker = self._field_unpacker(names)
        indexes = [physical_names.index(name) for name in names]
        for values in unpacker.iter_unpack(self.data):
            ordered = tuple(values[index] for index in indexes)
            yield ordered  # type: ignore[misc]


def parse_pointcloud2(payload: bytes) -> PointCloud2:
    reader = CdrReader(payload)
    stamp_sec = reader.i32()
    stamp_nanosec = reader.u32()
    frame_id = reader.string()
    height = reader.u32()
    width = reader.u32()
    field_count = reader.u32()
    fields = []
    for _ in range(field_count):
        fields.append(
            PointField(
                name=reader.string(),
                offset=reader.u32(),
                datatype=reader.u8(),
                count=reader.u32(),
            )
        )
    is_bigendian = bool(reader.u8())
    point_step = reader.u32()
    row_step = reader.u32()
    data_size = reader.u32()
    data = memoryview(payload)[reader.pos : reader.pos + data_size]
    reader.pos += data_size
    is_dense = bool(reader.u8())

    expected_size = height * row_step
    if len(data) != expected_size:
        raise ValueError(f"Point data size {len(data)} != expected {expected_size}")
    return PointCloud2(
        stamp_sec=stamp_sec,
        stamp_nanosec=stamp_nanosec,
        frame_id=frame_id,
        height=height,
        width=width,
        fields=tuple(fields),
        is_bigendian=is_bigendian,
        point_step=point_step,
        row_step=row_step,
        data=data,
        is_dense=is_dense,
    )


def iter_bag_messages(db3_path: str | Path) -> Iterator[tuple[int, PointCloud2]]:
    """Read PointCloud2 messages ordered by rosbag timestamp."""
    uri = f"file:{Path(db3_path).resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT messages.timestamp, messages.data
            FROM messages
            JOIN topics ON topics.id = messages.topic_id
            WHERE topics.type = 'sensor_msgs/msg/PointCloud2'
            ORDER BY messages.timestamp
            """
        )
        for bag_timestamp, payload in rows:
            yield bag_timestamp, parse_pointcloud2(payload)


def is_valid_xyz(x: float, y: float, z: float) -> bool:
    return math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and (x != 0 or y != 0 or z != 0)
