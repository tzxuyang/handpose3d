"""Quaternion and rotation primitives used by the VIO integration.

Convention: quaternions are ``(w, x, y, z)`` (Basalt/TUM order) unless a
function name says otherwise.  The foxglove messages use ``(x, y, z, w)`` and
are converted explicitly at the boundary (``quat_xyzw_to_wxyz``).

This module has no project dependencies so it can be unit tested in isolation.
"""

from __future__ import annotations

import math

import numpy as np


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    """Return ``quat`` normalised to unit length (identity when it is zero)."""
    quat = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def quat_slerp(quat0: np.ndarray, quat1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation of two ``(w, x, y, z)`` quaternions."""
    q0 = quat_normalize(quat0)
    q1 = quat_normalize(quat1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return quat_normalize(q0 + alpha * (q1 - q0))
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta0 * alpha
    sin_theta0 = math.sin(theta0)
    if abs(sin_theta0) < 1e-12:
        return q0
    return quat_normalize(
        math.sin(theta0 - theta) / sin_theta0 * q0
        + math.sin(theta) / sin_theta0 * q1
    )


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix of a ``(w, x, y, z)`` quaternion."""
    qw, qx, qy, qz = quat_normalize(np.asarray(quat_wxyz, dtype=float))
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert a foxglove-style ``(x, y, z, w)`` quaternion to ``(w, x, y, z)``."""
    qx, qy, qz, qw = np.asarray(quat_xyzw, dtype=float).reshape(4)
    return np.array([qw, qx, qy, qz], dtype=float)


def se3_row_to_pose(se3: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Split ``[x, y, z, qx, qy, qz, qw]`` into position and ``(w,x,y,z)`` quat."""
    values = np.asarray(se3, dtype=float).reshape(-1)
    if values.size != 7:
        raise ValueError(f"expected 7 values (x,y,z,qx,qy,qz,qw), got {values.size}")
    return values[:3].copy(), quat_xyzw_to_wxyz(values[3:])


def rotation_z(angle_deg: float) -> np.ndarray:
    """Rotation about +z by ``angle_deg`` degrees."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def rotation_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation of ``angle_deg`` about ``axis`` (need not be unit)."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle_deg) < 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / norm
    angle = math.radians(angle_deg)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
