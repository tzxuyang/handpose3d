"""Quality metrics for the fused trajectory and the world-frame output."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .frames import valid_point_mask
from .trajectory import PoseTrajectory


def trajectory_qc(
    trajectory: PoseTrajectory,
    timestamps_ns: np.ndarray | None = None,
) -> dict:
    """Summarise a trajectory: span, sample rate, gaps and query coverage."""
    times = np.asarray(trajectory.times_ns, dtype=np.int64)
    gaps = np.diff(times) if times.size > 1 else np.zeros(0, dtype=np.int64)
    report = {
        "source": trajectory.source,
        "anchor": trajectory.anchor,
        "yaw_deg": float(trajectory.yaw_deg),
        "samples": int(times.size),
        "first_timestamp_ns": int(times[0]),
        "last_timestamp_ns": int(times[-1]),
        "duration_s": float(trajectory.duration_s),
        "median_dt_ns": float(np.median(gaps)) if gaps.size else None,
        "max_gap_ns": int(gaps.max()) if gaps.size else None,
    }
    if timestamps_ns is not None and np.asarray(timestamps_ns).size:
        queries = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
        outside = trajectory.outside_span(queries)
        report.update(
            {
                "queries": int(queries.size),
                "queries_covered": int((~outside).sum()),
                "coverage": float((~outside).mean()),
                "queries_before_start": int((queries < trajectory.start_ns).sum()),
                "queries_after_end": int((queries > trajectory.end_ns).sum()),
            }
        )
    return report


def world_stability_qc(
    world_points: np.ndarray,
    timestamps_ns: np.ndarray,
) -> dict:
    """Frame-to-frame motion of the world-frame keypoints.

    ``world_points`` is shaped ``(frames, hands, keypoints, 3)``.  Only frames
    where the same keypoint is valid in both frames contribute; the metric is a
    cheap way to spot a broken transform (a static hand should stay put).
    """
    points = np.asarray(world_points, dtype=float)
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    if points.ndim < 2 or points.shape[0] < 2:
        return {"frames": int(points.shape[0]) if points.ndim else 0}

    valid = valid_point_mask(points)
    both = valid[1:] & valid[:-1]
    if not both.any():
        return {"frames": int(points.shape[0]), "valid_pairs": 0}

    delta = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    steps = delta[both]
    dt = np.abs(np.diff(timestamps_ns)) * 1e-9
    dt_pairs = np.broadcast_to(dt.reshape(-1, 1, 1), delta.shape)[both]
    speed = steps / np.maximum(dt_pairs, 1e-9)
    return {
        "frames": int(points.shape[0]),
        "valid_pairs": int(both.sum()),
        "step_mean_m": float(steps.mean()),
        "step_median_m": float(np.median(steps)),
        "step_p95_m": float(np.percentile(steps, 95)),
        "speed_median_m_s": float(np.median(speed)),
    }


def write_qc_json(path: str | os.PathLike, report: dict) -> Path:
    """Write the QC report next to the produced artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
    return path


def update_qc_json(path: str | os.PathLike, updates: dict) -> Path:
    """Merge ``updates`` into an existing QC report (creating it when absent)."""
    path = Path(path)
    existing: dict = {}
    if path.exists():
        try:
            with open(path, "r") as handle:
                existing = json.load(handle)
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(updates)
    return write_qc_json(path, existing)
