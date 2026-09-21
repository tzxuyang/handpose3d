"""Frame conventions for the hand keypoints and the rigid point transform.

The stereo pipeline stores keypoints after ``rotate_points_around_z(p, +90)``
(a ``(x right, y forward, z up)`` convention), while the camera calibration
``T_b_c`` and the VIO pose live in the device body frame (``(x forward, y left,
z up)``).  ``published_to_body`` / ``body_to_published`` convert between the
two; ``transform_points_to_world`` applies an interpolated trajectory pose.
"""

from __future__ import annotations

import numpy as np

from .rotations import rotation_z


def published_to_body(points: np.ndarray) -> np.ndarray:
    """Undo the ``rotate_points_around_z(+90)`` convention of the pipeline."""
    points = np.asarray(points, dtype=float)
    return points @ rotation_z(-90.0).T


def body_to_published(points: np.ndarray) -> np.ndarray:
    """Apply the ``rotate_points_around_z(+90)`` convention of the pipeline."""
    points = np.asarray(points, dtype=float)
    return points @ rotation_z(90.0).T


def valid_point_mask(points: np.ndarray) -> np.ndarray:
    """Mask of usable keypoints (finite and not the ``(-1, -1, -1)`` sentinel)."""
    points = np.asarray(points, dtype=float)
    return np.isfinite(points).all(axis=-1) & ~np.all(points == -1.0, axis=-1)


def transform_points_to_world(
    points_body: np.ndarray,
    rotations: np.ndarray,
    positions: np.ndarray,
    undo_z_rotation: bool = True,
    t_imu_body: np.ndarray | None = None,
) -> np.ndarray:
    """Map published-frame hand keypoints to the VIO world frame.

    ``handpose3d`` stores the triangulated points after rotating them 90 deg
    about z (``rotate_points_around_z(point_3d, 90)``), so that rotation is
    undone first before the trajectory pose is applied:
    ``p_world = T_w_imu(t) * T_imu_body * R_z(-90) * p_stored``.

    Kept as a module-level function for the CLI/mcap path; the inline path uses
    :meth:`vio.trajectory.PoseTrajectory.transform_points`.
    """
    points_body = np.asarray(points_body, dtype=float)
    valid = valid_point_mask(points_body)

    local = np.array(points_body, dtype=float)
    if undo_z_rotation:
        local = local @ rotation_z(-90.0).T
    if t_imu_body is not None:
        t_imu_body = np.asarray(t_imu_body, dtype=float).reshape(4, 4)
        local = local @ t_imu_body[:3, :3].T + t_imu_body[:3, 3]

    world = np.einsum("nij,nkj->nki", rotations, local) + positions[:, None, :]
    world[~valid] = -1.0
    return world
