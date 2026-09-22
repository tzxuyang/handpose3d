from __future__ import annotations

import math
import unittest

import numpy as np

from tests import SRC_DIR  # noqa: F401  (ensures `src` is importable)
from vio.rotations import (
    quat_normalize,
    quat_slerp,
    quat_wxyz_to_matrix,
    quat_xyzw_to_wxyz,
    rotation_axis_angle,
    rotation_z,
    se3_row_to_pose,
)


class RotationTests(unittest.TestCase):
    def test_quat_normalize_handles_zero(self) -> None:
        np.testing.assert_allclose(quat_normalize(np.zeros(4)), [1.0, 0.0, 0.0, 0.0])

    def test_quat_wxyz_to_matrix_identity(self) -> None:
        np.testing.assert_allclose(quat_wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0])), np.eye(3))

    def test_quat_wxyz_to_matrix_is_orthonormal(self) -> None:
        rotation = quat_wxyz_to_matrix(np.array([0.5, -0.5, 0.5, -0.5]))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_quat_slerp_midpoint_is_half_rotation(self) -> None:
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
        midpoint = quat_slerp(q0, q1, 0.5)
        expected = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
        np.testing.assert_allclose(midpoint, expected, atol=1e-12)

    def test_quat_slerp_takes_the_short_path(self) -> None:
        # The same small rotation, represented with a negative scalar part:
        # the interpolant must stay close to the identity instead of walking
        # the long way around.
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        angle = math.radians(10.0)
        q1 = -np.array([math.cos(angle), 0.0, 0.0, math.sin(angle)])
        midpoint = quat_slerp(q0, q1, 0.5)
        expected = np.array(
            [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
        )
        np.testing.assert_allclose(midpoint, expected, atol=1e-12)

    def test_quat_xyzw_round_trip(self) -> None:
        np.testing.assert_allclose(
            quat_xyzw_to_wxyz([0.1, 0.2, 0.3, 0.4]), [0.4, 0.1, 0.2, 0.3]
        )

    def test_se3_row_to_pose(self) -> None:
        position, quat = se3_row_to_pose([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9])
        np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(quat, [0.9, 0.1, 0.2, 0.3])

    def test_rotation_z_rotates_x_into_y(self) -> None:
        np.testing.assert_allclose(rotation_z(90.0) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(rotation_z(-90.0) @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-12)

    def test_rotation_axis_angle_matches_z_rotation(self) -> None:
        np.testing.assert_allclose(
            rotation_axis_angle(np.array([0.0, 0.0, 2.0]), 90.0), rotation_z(90.0), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
