from __future__ import annotations

import unittest

import numpy as np

from tests import SRC_DIR  # noqa: F401
from vio.frames import (
    body_to_published,
    published_to_body,
    transform_points_to_world,
    valid_point_mask,
)
from vio.trajectory import PoseTrajectory


class FrameTests(unittest.TestCase):
    def test_published_body_round_trip(self) -> None:
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        np.testing.assert_allclose(published_to_body(body_to_published(points)), points, atol=1e-12)

    def test_valid_point_mask(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0],
                [np.nan, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(valid_point_mask(points), [True, False, False, True])

    def test_transform_matches_pose_trajectory_api(self) -> None:
        """The CLI helper and the inline API must produce identical points."""
        rng = np.random.default_rng(7)
        frames = 5
        points = rng.normal(size=(frames, 2, 21, 3))
        points[1, 0, :, :] = -1.0
        points[3, 1, :5, :] = np.nan
        timestamps = np.arange(frames, dtype=np.int64) * 100

        times = np.array([0, 200, 400], dtype=np.int64)
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
        trajectory = PoseTrajectory(
            times, quats, positions, anchor="position", yaw_deg=30.0
        )

        inline = trajectory.transform_points(points, timestamps)

        rotations, anchored_positions = trajectory.relative(timestamps)
        from vio.rotations import rotation_axis_angle

        yaw = rotation_axis_angle(trajectory.vertical_axis(), trajectory.yaw_deg)
        reference = np.stack(
            [
                transform_points_to_world(
                    points[:, hand],
                    rotations,
                    anchored_positions,
                    undo_z_rotation=True,
                )
                for hand in range(points.shape[1])
            ],
            axis=1,
        )
        reference = np.einsum("ij,fmkj->fmki", yaw, reference)
        reference[~valid_point_mask(points)] = -1.0
        np.testing.assert_allclose(inline, reference, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
