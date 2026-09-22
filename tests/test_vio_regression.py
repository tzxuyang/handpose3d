"""Regression tests against the pre-refactor ``basalt_bridge`` implementation.

These tests need local artifacts that are not part of the repository (a legacy
copy of the bridge, a head-frame hand mcap, a Basalt trajectory and a raw
recording); they skip when the artifacts are absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import SRC_DIR  # noqa: F401
from vio.pipeline import transform_hand_mcap

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_BRIDGE = Path("/tmp/basalt_bridge_legacy.py")
HEAD_MCAP = REPO_ROOT / "mcap_output" / "hand_keypoints.mcap"
TRAJECTORY = REPO_ROOT / "basalt" / "trajectory.txt"
RAW_MCAP_CANDIDATES = (
    Path("/home/ubuntu/handpose3D_dataSet/groundTruth_data/ff9e3e1189504041b9ce21256925377f.mcap"),
    Path("/home/ubuntu/handpose3D_dataSet/dataset/origin_data_4.mcap"),
)


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("basalt_bridge_legacy", LEGACY_BRIDGE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses look the class' module up in sys.modules while decorating
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_world_points(path: Path) -> list[np.ndarray]:
    from mcap_utils import read_hand_json_3d, read_mcap_json

    frames = []
    for topic in ("/robot0/handtracking/left", "/robot0/handtracking/right"):
        messages = read_mcap_json(str(path), topic)
        frames.append(np.asarray([read_hand_json_3d(message) for message in messages]))
    return frames


class LegacyTransformRegression(unittest.TestCase):
    def setUp(self) -> None:
        missing = [p for p in (LEGACY_BRIDGE, HEAD_MCAP, TRAJECTORY) if not p.exists()]
        if missing:
            self.skipTest(f"artifacts missing: {missing}")

    def test_world_mcap_matches_legacy(self) -> None:
        legacy = load_legacy_module()
        with tempfile.TemporaryDirectory() as tmp:
            legacy_out = Path(tmp) / "legacy_world.mcap"
            new_out = Path(tmp) / "new_world.mcap"

            legacy.transform_hand_mcap(
                str(HEAD_MCAP),
                str(TRAJECTORY),
                str(legacy_out),
                undo_z_rotation=True,
                anchor="position",
                yaw_deg=89.8,
                verbose=False,
            )
            transform_hand_mcap(
                str(HEAD_MCAP),
                str(TRAJECTORY),
                str(new_out),
                undo_z_rotation=True,
                anchor="position",
                yaw_deg=89.8,
                verbose=False,
            )

            legacy_hands = read_world_points(legacy_out)
            new_hands = read_world_points(new_out)

        self.assertEqual(len(legacy_hands), len(new_hands))
        for hand_index, (legacy_hand, new_hand) in enumerate(zip(legacy_hands, new_hands)):
            self.assertEqual(legacy_hand.shape, new_hand.shape)
            np.testing.assert_allclose(
                new_hand,
                legacy_hand,
                atol=1e-9,
                err_msg=f"world keypoints differ on hand {hand_index}",
            )


class LegacyExportRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_mcap = next((p for p in RAW_MCAP_CANDIDATES if p.exists()), None)
        if LEGACY_BRIDGE.exists() is False or self.raw_mcap is None:
            self.skipTest("legacy bridge or raw recording not available")

    def test_dataset_export_matches_legacy(self) -> None:
        legacy = load_legacy_module()
        with tempfile.TemporaryDirectory() as tmp:
            legacy_dir = Path(tmp) / "legacy"
            new_dir = Path(tmp) / "new"

            legacy.export_euroc_dataset(
                str(self.raw_mcap),
                legacy_dir,
                str(REPO_ROOT / "configs" / "ego_config.json"),
                cam_ids=(1, 4),
                scale=0.5,
                max_frames=30,
                verbose=False,
            )
            from vio.dataset import export_euroc_dataset

            export_euroc_dataset(
                str(self.raw_mcap),
                new_dir,
                str(REPO_ROOT / "configs" / "ego_config.json"),
                cam_ids=(1, 4),
                scale=0.5,
                max_frames=30,
                verbose=False,
            )

            for relative in (
                "mav0/cam0/data.csv",
                "mav0/cam1/data.csv",
                "mav0/imu0/data.csv",
                "basalt_calib.json",
            ):
                legacy_file = legacy_dir / relative
                new_file = new_dir / relative
                self.assertTrue(legacy_file.exists(), relative)
                self.assertTrue(new_file.exists(), relative)
                self.assertEqual(
                    new_file.read_text(),
                    legacy_file.read_text(),
                    f"{relative} differs from the legacy export",
                )
            # frame files must exist for every cam0 row
            rows = (new_dir / "mav0" / "cam0" / "data.csv").read_text().strip().splitlines()
            for row in rows:
                name = row.split(",")[1]
                self.assertTrue((new_dir / "mav0" / "cam0" / "data" / name).exists(), name)
                self.assertTrue((new_dir / "mav0" / "cam1" / "data" / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
