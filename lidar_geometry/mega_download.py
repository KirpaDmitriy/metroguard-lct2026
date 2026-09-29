"""Download a public MEGA file using only stdlib plus the system OpenSSL."""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def decode_key(encoded: str) -> tuple[bytes, bytes]:
    raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    if len(raw) != 32:
        raise ValueError("Expected an eight-word MEGA public-file key")
    words = struct.unpack(">8I", raw)
    aes_key = struct.pack(
        ">4I", *(words[index] ^ words[index + 4] for index in range(4))
    )
    iv = struct.pack(">4I", words[4], words[5], 0, 0)
    return aes_key, iv


def download(public_link: str, output: Path, workers: int = 8) -> None:
    before_key, encoded_key = public_link.split("#", 1)
    handle = before_key.rstrip("/").rsplit("/", 1)[-1]
    payload = json.dumps([{"a": "g", "g": 1, "p": handle}]).encode()
    request = urllib.request.Request(
        "https://g.api.mega.co.nz/cs?id=0",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    metadata = json.loads(urllib.request.urlopen(request).read())[0]
    if isinstance(metadata, int):
        raise RuntimeError(f"MEGA API error {metadata}")
    aes_key, iv = decode_key(encoded_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    total_size = metadata["s"]
    part_size = math.ceil(total_size / workers)
    encrypted_parts = [
        output.with_suffix(output.suffix + f".enc.{index:02d}")
        for index in range(workers)
    ]

    def fetch(index: int) -> Path:
        start = index * part_size
        end = min(total_size, start + part_size) - 1
        expected = end - start + 1
        target = encrypted_parts[index]
        if target.exists() and target.stat().st_size == expected:
            return target
        request = urllib.request.Request(f"{metadata['g']}/{start}-{end}")
        with urllib.request.urlopen(request) as source, target.open("wb") as stream:
            while chunk := source.read(1024 * 1024):
                stream.write(chunk)
        if target.stat().st_size != expected:
            raise RuntimeError(
                f"Range {index} has {target.stat().st_size} bytes, expected {expected}"
            )
        print(json.dumps({"output": str(output), "range_complete": index}), flush=True)
        return target

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(workers)]
        for future in as_completed(futures):
            future.result()

    with temporary.open("wb") as target:
        process = subprocess.Popen(
            [
                "openssl",
                "enc",
                "-d",
                "-aes-128-ctr",
                "-K",
                aes_key.hex(),
                "-iv",
                iv.hex(),
            ],
            stdin=subprocess.PIPE,
            stdout=target,
        )
        assert process.stdin is not None
        for encrypted_part in encrypted_parts:
            with encrypted_part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    process.stdin.write(chunk)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("OpenSSL failed to decrypt the MEGA stream")
    temporary.replace(output)
    if output.stat().st_size != metadata["s"]:
        raise RuntimeError(
            f"Unexpected output size {output.stat().st_size} != {metadata['s']}"
        )
    for encrypted_part in encrypted_parts:
        encrypted_part.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_link")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download(args.public_link, args.output, args.workers)


if __name__ == "__main__":
    main()
