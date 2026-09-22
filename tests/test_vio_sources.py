from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tests import SRC_DIR  # noqa: F401
from vio.sources import EefPoseSource, TumFileSource

GEN_NEW_MCAP = Path(
    "/home/ubuntu/handpose3D_dataSet/dataset/gen_new/domestic_services/study/"
    "life_operations/fill_fountain_pen_with_ink/0615c14d5f71458182d747df425893ba.mcap"
)


class TumFileSourceTests(unittest.TestCase):
    def test_loads_existing_trajectory(self) -> None:
        path = Path(__file__).resolve().parents[1] / "basalt" / "trajectory.txt"
        if not path.exists():
            self.skipTest("basalt/trajectory.txt not available")
        trajectory = TumFileSource(path, anchor="position", yaw_deg=89.8).load()
        self.assertGreater(trajectory.times_ns.size, 100)
        self.assertGreater(trajectory.duration_s, 1.0)
        self.assertTrue(np.all(np.diff(trajectory.times_ns) > 0))


class EefPoseSourceTests(unittest.TestCase):
    def test_loads_device_world_poses(self) -> None:
        if not GEN_NEW_MCAP.exists():
            self.skipTest("gen_new recording with /robot0/vio/eef_pose not available")
        trajectory = EefPoseSource(GEN_NEW_MCAP).load()
        self.assertGreater(trajectory.times_ns.size, 100)
        self.assertGreater(trajectory.duration_s, 10.0)
        self.assertTrue(np.all(np.diff(trajectory.times_ns) > 0))
        # the device's world frame starts at the recording origin
        self.assertLess(float(np.abs(trajectory.positions[0]).max()), 1e-3)
        self.assertLess(float(np.abs(trajectory.positions).max()), 50.0)


if __name__ == "__main__":
    unittest.main()
