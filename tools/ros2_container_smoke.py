#!/usr/bin/env python3
"""Run and compare two clean MetroGuard ROS 2 container sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "tools" / "ros2_capture.py"


def command(*args: str, timeout: float | None = None) -> str:
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def remove_containers(names: list[str]) -> None:
    subprocess.run(
        ["docker", "rm", "-f", *names],
        check=False,
        capture_output=True,
        text=True,
    )


def capture_run(
    run_index: int,
    network: str,
    image: str,
    bag_dir: Path,
    count: int,
    timeout: float,
) -> list[dict]:
    prefix = f"metro-guard-smoke-{run_index}"
    detector = f"{prefix}-detector"
    capture = f"{prefix}-capture"
    player = f"{prefix}-player"
    names = [player, capture, detector]
    remove_containers(names)
    try:
        command(
            "docker", "run", "-d", "--name", detector,
            "--network", network, "-e", "ROS_DOMAIN_ID=73", image,
        )
        command(
            "docker", "run", "-d", "--name", capture,
            "--network", network, "-e", "ROS_DOMAIN_ID=73",
            "-v", f"{CAPTURE}:/tmp/ros2_capture.py:ro", image,
            "python3", "/tmp/ros2_capture.py",
            "--count", str(count), "--timeout", str(timeout),
        )
        command(
            "docker", "run", "-d", "--name", player,
            "--network", network, "-e", "ROS_DOMAIN_ID=73",
            "-v", f"{bag_dir.resolve()}:/bag:ro", image,
            "ros2", "bag", "play", "/bag",
            "--disable-keyboard-controls", "--delay", "5",
            "--read-ahead-queue-size", "10", "--rate", "1.0",
        )
        exit_code = command("docker", "wait", capture, timeout=timeout + 30)
        logs = command("docker", "logs", capture)
        if exit_code != "0":
            raise RuntimeError(f"capture exited with {exit_code}: {logs}")
        payloads = json.loads(logs.splitlines()[-1])
        if len(payloads) != count:
            raise RuntimeError(f"captured {len(payloads)}/{count} decisions")
        return payloads
    finally:
        remove_containers(names)


def stable_payload(payload: dict) -> dict:
    result = dict(payload)
    result.pop("latency_ms", None)
    return result


def latency_summary(payloads: list[dict]) -> dict:
    values = [float(item["latency_ms"]) for item in payloads]
    ordered = sorted(values)
    p95_index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("--image", default="metro-guard:mvp")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "lidar_geometry/artifacts/ros2_container_smoke.json",
    )
    args = parser.parse_args()
    if not (args.bag_dir / "metadata.yaml").is_file():
        parser.error("bag_dir must contain metadata.yaml")

    network = "metro-guard-repro-smoke"
    subprocess.run(
        ["docker", "network", "rm", network],
        check=False,
        capture_output=True,
        text=True,
    )
    command("docker", "network", "create", network)
    try:
        runs = [
            capture_run(index, network, args.image, args.bag_dir, args.count, args.timeout)
            for index in (1, 2)
        ]
    finally:
        subprocess.run(
            ["docker", "network", "rm", network],
            check=False,
            capture_output=True,
            text=True,
        )

    stable_runs = [[stable_payload(item) for item in run] for run in runs]
    deterministic = stable_runs[0] == stable_runs[1]
    report = {
        "status": "PASS" if deterministic else "FAIL",
        "image": args.image,
        "image_id": command("docker", "image", "inspect", args.image, "--format", "{{.Id}}"),
        "bag_dir": str(args.bag_dir),
        "count_per_run": args.count,
        "playback_rate": 1.0,
        "runs": [
            {
                "states": [item["state"] for item in run],
                "stamps": [item["stamp"] for item in run],
                "latency": latency_summary(run),
            }
            for run in runs
        ],
        "stable_payloads_equal": deterministic,
        "comparison_excludes": ["latency_ms"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not deterministic:
        sys.exit(1)


if __name__ == "__main__":
    main()
