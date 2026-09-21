"""Reading raw recordings: topics, camera info, H.264 frames and IMU samples."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from ._deps import extract_h264_data, read_mcap_protobuf


def load_ego_config(config_path: str | os.PathLike = "./configs/ego_config.json") -> dict:
    """Read the topic list used by the handpose3d pipeline."""
    with open(config_path, "r") as handle:
        return json.load(handle)


def list_mcap_topics(mcap_path: str | os.PathLike) -> list[str]:
    """Return every topic present in an mcap file."""
    with open(mcap_path, "rb") as handle:
        summary = make_reader(handle).get_summary()
    if summary is None or not summary.channels:
        return []
    return sorted({channel.topic for channel in summary.channels.values()})


def missing_input_hint(mcap_path: str | os.PathLike, topic: str) -> str:
    """Explain that the raw recording (cameras + IMU) is the expected input."""
    try:
        topics = list_mcap_topics(mcap_path)
    except Exception:  # pragma: no cover - only used to build a message
        topics = []

    cameras = [name for name in topics if "camera" in name and "compressed" in name]
    hands = [name for name in topics if "handtracking" in name]
    if cameras:
        return f"topic {topic} not found in {mcap_path}; cameras available: {cameras}"

    hint = (
        f"{mcap_path} contains no camera images, so Basalt cannot run on it.\n"
        "Pass the RAW recording (the mcap you feed to `main.py --mode process`, which holds the\n"
        "six /robot0/sensor/camera*/compressed topics and /robot0/sensor/imu), and point\n"
        "`--hand_mcap` at the 3D hand keypoints from that same recording."
    )
    if hands:
        hint += f"\nThe file looks like a hand keypoints product (topics: {hands})."
    if topics:
        hint += f"\nTopics found: {topics}"
    return hint


def read_camera_info(mcap_path: str | os.PathLike, topic: str) -> dict:
    """Return the first ``camera_info`` message of ``topic`` as a plain dict."""
    with open(mcap_path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, _channel, _message, proto in reader.iter_decoded_messages(topics=[topic]):
            if proto is None:
                continue
            return {
                "topic": topic,
                "frame_id": str(getattr(proto, "frame_id", "")),
                "width": int(proto.width),
                "height": int(proto.height),
                "K": [float(value) for value in proto.K],
                "D": [float(value) for value in proto.D],
                "distortion_model": str(proto.distortion_model),
                "T_b_c": [float(value) for value in proto.T_b_c],
            }
    raise RuntimeError(f"no camera_info message found on {topic} in {mcap_path}")


def read_camera_messages(
    mcap_path: str | os.PathLike, topic: str
) -> tuple[list[int], list[bytes]]:
    """Collect ``(header timestamps in ns, H.264 payloads)`` of one camera topic."""
    timestamps: list[int] = []
    payloads: list[bytes] = []
    with open(mcap_path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, _channel, message, proto in reader.iter_decoded_messages(topics=[topic]):
            if proto is None:
                continue
            payload = extract_h264_data(message.data)
            if not payload:
                continue
            timestamps.append(int(proto.header.timestamp))
            payloads.append(payload)
    if not timestamps:
        raise RuntimeError(missing_input_hint(mcap_path, topic))
    return timestamps, payloads


def read_imu_samples(
    mcap_path: str | os.PathLike, imu_topic: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(timestamps_ns, gyro_rad_s, accel_in_g)`` of an IMU topic."""
    messages = read_mcap_protobuf(str(mcap_path), imu_topic)
    times, gyro, accel = [], [], []
    for message in messages:
        linear = message.get("linear_acceleration") or message.get("linear_acceleraion")
        angular = message["angular_velocity"]
        times.append(int(message["header"]["timestamp"]))
        gyro.append([float(angular["x"]), float(angular["y"]), float(angular["z"])])
        accel.append([float(linear["x"]), float(linear["y"]), float(linear["z"])])
    return (
        np.asarray(times, dtype=np.int64),
        np.asarray(gyro, dtype=float),
        np.asarray(accel, dtype=float),
    )


def decode_h264_sequence(
    payloads: list[bytes],
    out_dir: Path,
    tag: str,
    scale: float = 1.0,
    gray: bool = True,
    probe_messages: int = 4,
    verbose: bool = True,
) -> tuple[list[Path], int]:
    """Decode concatenated H.264 access units into a PNG sequence.

    Returns ``(frames, shift)``.  ``shift`` is the number of leading messages
    that produced no frame: these recordings start on a P-slice whose
    reference frame is not in the stream, so ffmpeg drops it and every later
    frame would otherwise be paired with the previous timestamp.  The caller
    must pair ``frames[i]`` with ``messages[i + shift]``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h264_path = out_dir / f".{tag}.h264"
    with open(h264_path, "wb") as handle:
        for payload in payloads:
            handle.write(payload)

    filters = []
    if abs(scale - 1.0) > 1e-9:
        filters.append(f"scale=iw*{scale}:ih*{scale}:flags=area")
    if gray:
        filters.append("format=gray")

    def run(count: int, pattern: str) -> list[Path]:
        probe_path = out_dir / f".{tag}_probe.h264"
        probe_path.write_bytes(b"".join(payloads[:count]))
        cmd = ["ffmpeg", "-y", "-f", "h264", "-i", str(probe_path), "-vsync", "0"]
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd.append(str(out_dir / pattern))
        result = subprocess.run(cmd, capture_output=True, text=True)
        probe_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {tag}: {result.stderr[-500:]}")
        return sorted(out_dir.glob(pattern.replace("%06d", "*")))

    probe_count = min(probe_messages, len(payloads))
    shift = max(0, probe_count - len(run(probe_count, f".{tag}_probe_%06d.png")))
    for stale in out_dir.glob(f".{tag}_probe_*.png"):
        stale.unlink()

    frames = run(len(payloads), f"{tag}_%06d.png")
    frames = [path for path in frames if path.name.startswith(f"{tag}_")]

    expected = len(payloads) - shift
    if len(frames) != expected and verbose:
        print(
            f"  ! {tag}: decoded {len(frames)} frames for {len(payloads)} messages "
            f"(expected {expected}); using the common prefix"
        )
    return frames, shift


def write_csv(path: Path, rows: list[list[object]]) -> None:
    """Write a simple comma-separated table (used for the EuRoC csv files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(",".join(str(value) for value in row))
            handle.write("\n")
