"""TUM trajectories, interpolation, anchoring and the ``PoseTrajectory`` API.

Everything in this module is pure numerics: no mcap, no subprocess, no
Basalt.  ``PoseTrajectory`` is the neutral interface the rest of the pipeline
(and the inline hand transform) consumes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rotations import (
    quat_slerp,
    quat_wxyz_to_matrix,
    rotation_axis_angle,
    rotation_z,
)


def load_tum_trajectory(
    path: str | os.PathLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a TUM trajectory; returns ``(timestamps_ns, quats_wxyz, positions)``."""
    times, quats, positions = [], [], []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 8:
                continue
            timestamp, tx, ty, tz, qx, qy, qz, qw = (float(value) for value in parts)
            times.append(timestamp * 1e9)
            positions.append([tx, ty, tz])
            quats.append([qw, qx, qy, qz])
    if not times:
        raise RuntimeError(f"no poses parsed from {path}")
    return (
        np.asarray(times, dtype=np.int64),
        np.asarray(quats, dtype=float),
        np.asarray(positions, dtype=float),
    )


def interpolate_poses(
    query_ns: np.ndarray,
    traj_ns: np.ndarray,
    traj_quats_wxyz: np.ndarray,
    traj_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the trajectory at ``query_ns``; returns ``(rotations, positions)``.

    Queries outside the trajectory span are clamped to the nearest sample
    (callers report that case through :func:`outside_span_mask`).
    """
    query_ns = np.asarray(query_ns, dtype=np.int64)
    traj_ns = np.asarray(traj_ns, dtype=np.int64)
    traj_quats_wxyz = np.asarray(traj_quats_wxyz, dtype=float)
    traj_positions = np.asarray(traj_positions, dtype=float)

    if traj_ns.size == 1:
        rotation = quat_wxyz_to_matrix(traj_quats_wxyz[0])
        return np.repeat(rotation[None], query_ns.size, axis=0), np.repeat(
            traj_positions[:1], query_ns.size, axis=0
        )

    index = np.clip(np.searchsorted(traj_ns, query_ns, side="right"), 1, traj_ns.size - 1)
    span = np.maximum(traj_ns[index] - traj_ns[index - 1], 1)
    alpha = np.clip((query_ns - traj_ns[index - 1]) / span, 0.0, 1.0)

    rotations = np.zeros((query_ns.size, 3, 3), dtype=float)
    positions = np.zeros((query_ns.size, 3), dtype=float)
    for i in range(query_ns.size):
        a = float(alpha[i])
        rotations[i] = quat_wxyz_to_matrix(
            quat_slerp(traj_quats_wxyz[index[i] - 1], traj_quats_wxyz[index[i]], a)
        )
        positions[i] = (
            (1.0 - a) * traj_positions[index[i] - 1] + a * traj_positions[index[i]]
        )
    return rotations, positions


def reanchor_poses(
    rotations: np.ndarray,
    positions: np.ndarray,
    anchor_rotation: np.ndarray,
    anchor_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express poses relative to an anchor, i.e. ``T_anchor^{-1} * T(t)``.

    This is the ``anchor="pose"`` convention: the origin *and* the axes are
    taken from the pose at the start of the episode, so the output frame tilts
    with however the wearer was oriented when the recording started.  The
    default ``anchor="position"`` only moves the origin and keeps the
    gravity-aligned axes, which is what makes a yaw correction meaningful.
    """
    anchored_positions = (np.asarray(positions, dtype=float) - anchor_position) @ anchor_rotation
    anchored_rotations = np.einsum(
        "ij,njk->nik",
        np.asarray(anchor_rotation, dtype=float).T,
        np.asarray(rotations, dtype=float),
    )
    return anchored_rotations, anchored_positions


def outside_span_mask(timestamps_ns: np.ndarray, trajectory_ns: np.ndarray) -> np.ndarray:
    """Boolean mask of query timestamps outside the trajectory's time span."""
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    trajectory_ns = np.asarray(trajectory_ns, dtype=np.int64)
    return (timestamps_ns < trajectory_ns[0]) | (timestamps_ns > trajectory_ns[-1])


@dataclass
class PoseTrajectory:
    """A pose trajectory in the VIO world frame with a chosen output convention.

    Parameters
    ----------
    times_ns, quats_wxyz, positions:
        raw trajectory samples (TUM order, quaternions ``(w, x, y, z)``).
    anchor:
        ``"position"`` keeps the gravity-aligned axes and moves the origin to
        the start pose; ``"pose"`` additionally takes the start orientation.
    yaw_deg:
        rotation about the vertical (gravity) axis applied to the world points
        after anchoring, used to line Basalt's world up with the robot world.
    t_imu_body:
        optional static 4x4 transform from the camera-rig/body frame to the
        VIO/IMU frame; applied to the points before the trajectory pose.
    source:
        free-form provenance string recorded in the QC report.
    """

    times_ns: np.ndarray
    quats_wxyz: np.ndarray
    positions: np.ndarray
    anchor: str = "position"
    yaw_deg: float = 0.0
    t_imu_body: Optional[np.ndarray] = None
    source: str = ""

    def __post_init__(self) -> None:
        self.times_ns = np.asarray(self.times_ns, dtype=np.int64).reshape(-1)
        self.quats_wxyz = np.asarray(self.quats_wxyz, dtype=float).reshape(-1, 4)
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        if not (self.times_ns.size == self.quats_wxyz.shape[0] == self.positions.shape[0]):
            raise ValueError("times, quaternions and positions must have the same length")
        if self.times_ns.size == 0:
            raise ValueError("a PoseTrajectory needs at least one sample")
        order = np.argsort(self.times_ns)
        self.times_ns = self.times_ns[order]
        self.quats_wxyz = self.quats_wxyz[order]
        self.positions = self.positions[order]
        if self.anchor not in ("position", "pose"):
            raise ValueError(f"unknown anchor mode {self.anchor!r}")
        if self.t_imu_body is not None:
            self.t_imu_body = np.asarray(self.t_imu_body, dtype=float).reshape(4, 4)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_tum(cls, path: str | os.PathLike, **kwargs) -> "PoseTrajectory":
        times, quats, positions = load_tum_trajectory(path)
        return cls(times, quats, positions, **kwargs)

    # -- queries -----------------------------------------------------------
    @property
    def start_ns(self) -> int:
        return int(self.times_ns[0])

    @property
    def end_ns(self) -> int:
        return int(self.times_ns[-1])

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) * 1e-9

    def anchor_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rotation, position)`` of the first sample."""
        return quat_wxyz_to_matrix(self.quats_wxyz[0]), self.positions[0].copy()

    def relative(
        self, query_ns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated poses expressed relative to the episode anchor."""
        rotations, positions = interpolate_poses(
            query_ns, self.times_ns, self.quats_wxyz, self.positions
        )
        anchor_rotation, anchor_position = self.anchor_pose()
        if self.anchor == "pose":
            return reanchor_poses(rotations, positions, anchor_rotation, anchor_position)
        return rotations, positions - anchor_position

    def vertical_axis(self) -> np.ndarray:
        """World-frame gravity axis used for the optional yaw correction."""
        if self.anchor == "pose":
            anchor_rotation, _ = self.anchor_pose()
            return anchor_rotation.T @ np.array([0.0, 0.0, 1.0])
        return np.array([0.0, 0.0, 1.0])

    def outside_span(self, timestamps_ns: np.ndarray) -> np.ndarray:
        return outside_span_mask(timestamps_ns, self.times_ns)

    # -- transformation ----------------------------------------------------
    def transform_points(
        self,
        points: np.ndarray,
        timestamps_ns: np.ndarray,
        *,
        undo_published_rotation: bool = True,
    ) -> np.ndarray:
        """Map body/published-frame keypoints to the configured world frame.

        ``points`` has shape ``(frames, ..., 3)`` and ``timestamps_ns`` one
        timestamp per frame.  Invalid points (any ``-1``/non-finite value) are
        passed through untouched.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError(f"expected points shaped (frames, ..., 3), got {points.shape}")
        timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
        if timestamps_ns.size != points.shape[0]:
            raise ValueError(
                f"got {timestamps_ns.size} timestamps for {points.shape[0]} frames"
            )

        valid = np.isfinite(points).all(axis=-1) & ~np.all(points == -1.0, axis=-1)
        local = points.reshape(points.shape[0], -1, 3)
        if undo_published_rotation:
            local = local @ rotation_z(-90.0).T
        if self.t_imu_body is not None:
            local = (
                local @ self.t_imu_body[:3, :3].T + self.t_imu_body[:3, 3]
            )

        rotations, positions = self.relative(timestamps_ns)
        world = np.einsum("fij,fkj->fki", rotations, local) + positions[:, None, :]

        if abs(self.yaw_deg) > 1e-9:
            yaw_rotation = rotation_axis_angle(self.vertical_axis(), self.yaw_deg)
            world = np.einsum("ij,fkj->fki", yaw_rotation, world)

        world = world.reshape(points.shape)
        world[~valid] = -1.0
        return world
