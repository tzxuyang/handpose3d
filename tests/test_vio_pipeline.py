from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from tests import SRC_DIR  # noqa: F401
from vio.config import VioConfig
from vio.pipeline import start_vio_task
from vio.qc import update_qc_json


TUM_LINE = "1.000000000 0 0 0 0 0 0 1\n"


class VioConfigTests(unittest.TestCase):
    def test_from_ego_config_reads_vio_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ego_config.json"
            path.write_text(json.dumps({"vio": {"cam0": 0, "yaw_deg": 12.5, "enabled": False}}))
            config = VioConfig.from_ego_config(path, mcap_path="/tmp/raw.mcap")
            self.assertEqual(config.cam0, 0)
            self.assertAlmostEqual(config.yaw_deg, 12.5)
            self.assertFalse(config.enabled)
            self.assertEqual(config.mcap_path, "/tmp/raw.mcap")
            self.assertEqual(config.ego_config_path, str(path))

    def test_from_ego_config_ignores_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ego_config.json"
            path.write_text(json.dumps({"vio": {"does_not_exist": 1, "cam1": 3}}))
            config = VioConfig.from_ego_config(path)
            self.assertEqual(config.cam1, 3)

    def test_validate_rejects_same_cameras(self) -> None:
        config = VioConfig(mcap_path="raw.mcap", cam0=1, cam1=1)
        with self.assertRaises(ValueError):
            config.validate()

    def test_derived_paths(self) -> None:
        config = VioConfig(mcap_path="raw.mcap", output_dir="/tmp/vio-out")
        self.assertEqual(config.trajectory_path, Path("/tmp/vio-out/trajectory.txt"))
        self.assertEqual(config.dataset_dir, Path("/tmp/vio-out/dataset"))
        self.assertEqual(config.resolved_qc_path, Path("/tmp/vio-out/qc.json"))
        config.fmt = "csv"
        self.assertEqual(config.trajectory_path, Path("/tmp/vio-out/trajectory.csv"))


class QcReportTests(unittest.TestCase):
    def test_update_qc_json_merges_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qc.json"
            path.write_text(json.dumps({"trajectory": {"samples": 10}}))
            update_qc_json(path, {"frames": 3, "world_stability": {"step_mean_m": 0.01}})
            report = json.loads(path.read_text())
            self.assertEqual(report["trajectory"]["samples"], 10)
            self.assertEqual(report["frames"], 3)
            self.assertIn("world_stability", report)

    def test_update_qc_json_ignores_corrupt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qc.json"
            path.write_text("{not json")
            update_qc_json(path, {"frames": 1})
            self.assertEqual(json.loads(path.read_text()), {"frames": 1})


class RunVioTests(unittest.TestCase):
    def test_disabled_config_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = VioConfig(mcap_path=Path(tmp) / "raw.mcap", enabled=False)
            result = start_vio_task(config).get(timeout=10)
            self.assertEqual(result.status, "skipped")
            self.assertFalse(result.ok)

    def test_cached_trajectory_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.mcap"
            raw.write_bytes(b"not really an mcap")
            time.sleep(0.01)
            (root / "trajectory.txt").write_text(TUM_LINE)

            config = VioConfig(mcap_path=str(raw), output_dir=str(root))
            result = start_vio_task(config).get(timeout=10)

            self.assertEqual(result.status, "ok", result.message)
            self.assertTrue(result.qc.get("cached"))
            self.assertIsNotNone(result.trajectory)
            np.testing.assert_allclose(result.trajectory.positions, [[0.0, 0.0, 0.0]])

    def test_missing_binary_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.mcap"
            raw.write_bytes(b"not really an mcap")
            config = VioConfig(
                mcap_path=str(raw),
                output_dir=str(root),
                skip_export=True,
                skip_run=False,
                reuse_trajectory=False,
                basalt_bin=str(root / "does-not-exist" / "basalt_vio"),
            )
            result = start_vio_task(config).get(timeout=30)
            self.assertEqual(result.status, "failed")
            self.assertIn("basalt_vio", result.message)

    def test_stale_trajectory_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trajectory.txt").write_text(TUM_LINE)
            time.sleep(0.01)
            raw = root / "raw.mcap"
            raw.write_bytes(b"not really an mcap")
            config = VioConfig(
                mcap_path=str(raw),
                output_dir=str(root),
                reuse_trajectory=True,
                skip_export=True,
                basalt_bin=str(root / "missing" / "basalt_vio"),
            )
            result = start_vio_task(config).get(timeout=30)
            # the trajectory predates the recording, so a run is attempted and fails
            self.assertEqual(result.status, "failed")


if __name__ == "__main__":
    unittest.main()
