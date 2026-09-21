from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import SRC_DIR  # noqa: F401

from handpose3d import _resolve_trajectory, _write_world_outputs
from mcap_utils import read_hand_json_3d, read_mcap_json, write_2d_hand_keypoints_mcap
from vio.pipeline import VioResult
from vio.trajectory import PoseTrajectory


class _StubTask:
    def __init__(self, result: VioResult):
        self._result = result

    def get(self, timeout=None) -> VioResult:  # noqa: ARG002
        return self._result


def _trajectory() -> PoseTrajectory:
    times = np.array([0, 100, 200], dtype=np.int64)
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    return PoseTrajectory(times, quats, positions, anchor="position")


class ResolveTrajectoryTests(unittest.TestCase):
    def test_not_requested(self) -> None:
        trajectory, status = _resolve_trajectory(None, None)
        self.assertIsNone(trajectory)
        self.assertEqual(status, "not requested")

    def test_provided_directly(self) -> None:
        expected = _trajectory()
        trajectory, status = _resolve_trajectory(None, expected)
        self.assertIs(trajectory, expected)
        self.assertEqual(status, "provided")

    def test_successful_task(self) -> None:
        expected = _trajectory()
        task = _StubTask(VioResult(status="ok", message="ran", trajectory=expected))
        trajectory, status = _resolve_trajectory(task, None)
        self.assertIs(trajectory, expected)
        self.assertIn("ok", status)

    def test_failed_task_falls_back(self) -> None:
        task = _StubTask(VioResult(status="failed", message="basalt missing"))
        trajectory, status = _resolve_trajectory(task, None)
        self.assertIsNone(trajectory)
        self.assertIn("failed", status)


class WriteWorldOutputsTests(unittest.TestCase):
    def test_world_mcap_contains_transformed_hands_and_2d_topics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames, hands, points = 3, 2, 21
            timestamps = np.array([0, 100, 200], dtype=np.int64)

            hand_2d_paths = []
            for camera in range(6):
                path = root / f"hand_2d_cam{camera}.mcap"
                keypoints = np.zeros((frames, hands, points, 2))
                keypoints[..., 0] = 10.0 + camera
                keypoints[..., 1] = 20.0
                write_2d_hand_keypoints_mcap(
                    keypoints,
                    [int(value) for value in timestamps],
                    f"/robot0/sensor/camera{camera}/pre/hand_keypoints2d",
                    str(path),
                )
                hand_2d_paths.append(str(path))

            head_points = np.zeros((frames, hands, points, 3))
            head_points[..., 0] = 0.0
            head_points[..., 1] = 1.0  # published-frame y
            head_points[1, 0, :, :] = -1.0  # invalid frame must stay invalid

            trajectory = _trajectory()
            expected = trajectory.transform_points(head_points, timestamps)

            qc = _write_world_outputs(
                head_points,
                timestamps,
                trajectory,
                hand_2d_paths,
                str(root / "hand_keypoints_3d_world.mcap"),
                str(root / "hand_keypoints_world.mcap"),
            )

            world_left = read_mcap_json(
                str(root / "hand_keypoints_world.mcap"), "/robot0/handtracking/left"
            )
            self.assertEqual(len(world_left), frames)
            for index, message in enumerate(world_left):
                np.testing.assert_allclose(
                    np.asarray(read_hand_json_3d(message)),
                    expected[index, 0],
                    atol=1e-9,
                )

            topics = {
                message["topic"]
                for message in read_mcap_json(str(root / "hand_keypoints_world.mcap"))
            }
            self.assertIn("/robot0/sensor/camera0/pre/hand_keypoints2d", topics)
            self.assertIn("/robot0/handtracking/right", topics)
            self.assertEqual(qc["frames"], frames)
            self.assertIn("world_stability", qc)


if __name__ == "__main__":
    unittest.main()
