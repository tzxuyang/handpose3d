from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import SRC_DIR  # noqa: F401
from vio.trajectory import (
    PoseTrajectory,
    interpolate_poses,
    load_tum_trajectory,
    outside_span_mask,
    reanchor_poses,
)


TUM_TEMPLATE = (
    "# timestamp tx ty tz qx qy qz qw\n"
    "1.000000000 {tx} 0 0 0 0 0 1\n"
    "1.100000000 {tx} 0 0 0 0 0 1\n"
    "1.200000000 {tx} 0 0 0 0 0 1\n"
)


class TrajectoryTests(unittest.TestCase):
    def _write_tum(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_load_tum_trajectory_round_trip(self) -> None:
        path = self._write_tum(TUM_TEMPLATE.format(tx="1.5"))
        times, quats, positions = load_tum_trajectory(path)
        np.testing.assert_array_equal(times, [1_000_000_000, 1_100_000_000, 1_200_000_000])
        np.testing.assert_allclose(positions[:, 0], [1.5, 1.5, 1.5])
        np.testing.assert_allclose(quats, np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))

    def test_interpolate_between_samples(self) -> None:
        times = np.array([0, 100])
        quats = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
        positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        rotations, out = interpolate_poses(np.array([50]), times, quats, positions)
        np.testing.assert_allclose(out, [[0.5, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-12)

    def test_interpolate_clamps_outside_span(self) -> None:
        times = np.array([0, 100])
        quats = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
        positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        _, out = interpolate_poses(np.array([-50, 150]), times, quats, positions)
        np.testing.assert_allclose(out, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-12)

    def test_reanchor_pose_convention(self) -> None:
        rotations = np.tile(np.eye(3), (2, 1, 1))
        positions = np.array([[1.0, 0, 0], [2.0, 0, 0]])
        anchor_rotation = np.eye(3)
        anchor_position = np.array([1.0, 0.0, 0.0])
        out_rotations, out_positions = reanchor_poses(
            rotations, positions, anchor_rotation, anchor_position
        )
        np.testing.assert_allclose(out_rotations, rotations)
        np.testing.assert_allclose(out_positions, [[0.0, 0, 0], [1.0, 0, 0]])

    def test_outside_span_mask(self) -> None:
        mask = outside_span_mask(np.array([-1, 5, 11]), np.array([0, 10]))
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_pose_trajectory_position_anchor(self) -> None:
        times = np.array([0, 100])
        positions = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]])
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        trajectory = PoseTrajectory(times, quats, positions, anchor="position")
        rotations, anchored = trajectory.relative(np.array([0, 100]))
        np.testing.assert_allclose(anchored, [[0, 0, 0], [0, 0, 1]], atol=1e-12)
        np.testing.assert_allclose(rotations, np.tile(np.eye(3), (2, 1, 1)), atol=1e-12)

    def test_pose_trajectory_pose_anchor_removes_start_rotation(self) -> None:
        yaw_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        quat = np.array([0.5, 0.0, 0.0, 0.5])  # (w,x,y,z) = 90 deg about z
        times = np.array([0, 100])
        trajectory = PoseTrajectory(
            times,
            np.tile(quat, (2, 1)),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            anchor="pose",
        )
        rotations, positions = trajectory.relative(np.array([0]))
        np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(trajectory.vertical_axis(), yaw_90.T @ [0, 0, 1], atol=1e-12)

    def test_transform_points_yaw_and_invalid_passthrough(self) -> None:
        times = np.array([0, 100])
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
        positions = np.zeros((2, 3))
        trajectory = PoseTrajectory(times, quats, positions, yaw_deg=90.0)
        points = np.zeros((2, 1, 1, 3))
        points[0, 0, 0] = [1.0, 0.0, 0.0]
        points[1, 0, 0] = [-1.0, -1.0, -1.0]
        world = trajectory.transform_points(
            points, np.array([0, 100]), undo_published_rotation=False
        )
        np.testing.assert_allclose(world[0, 0, 0], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(world[1, 0, 0], [-1.0, -1.0, -1.0])

    def test_transform_points_undoes_published_rotation(self) -> None:
        times = np.array([0])
        trajectory = PoseTrajectory(
            times, np.array([[1.0, 0.0, 0.0, 0.0]]), np.zeros((1, 3))
        )
        published = np.zeros((1, 1, 1, 3))
        published[0, 0, 0] = [0.0, 1.0, 0.0]  # published frame
        body = trajectory.transform_points(published, np.array([0]))
        np.testing.assert_allclose(body[0, 0, 0], [1.0, 0.0, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
