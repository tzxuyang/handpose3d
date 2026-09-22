"""Wiring test for ``main.run_process`` (no MediaPipe / no Basalt invoked)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import SRC_DIR  # noqa: F401

import main as main_module
from vio.config import VioConfig


class _StubTask:
    def get(self, timeout=None):  # pragma: no cover - never called in this test
        raise AssertionError("the fused pipeline should not resolve the task here")


class RunProcessWiringTests(unittest.TestCase):
    def test_process_starts_vio_and_passes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mcap = Path(tmp) / "raw.mcap"
            mcap.write_bytes(b"placeholder")
            world_mcap = str(Path(tmp) / "world.mcap")
            config = main_module.Config(
                mcap_path=str(mcap),
                mode="process",
                world_mcap_path=world_mcap,
            )

            captured: dict = {}

            def fake_start_vio(vio_config: VioConfig):
                captured["vio_config"] = vio_config
                return _StubTask()

            def fake_handpose3d(*args, **kwargs):
                captured["handpose_args"] = args
                captured["handpose_kwargs"] = kwargs
                return {"head_mcap": args[1], "world_mcap": kwargs["world_mcap_path"]}

            with mock.patch.object(main_module, "start_vio_task", fake_start_vio), mock.patch.object(
                main_module, "readmcap", return_value=(None, [0, 1, 2], 10, 10)
            ), mock.patch.object(main_module, "handpose3d", fake_handpose3d):
                summary = main_module.run_process(config)

            vio_config = captured["vio_config"]
            self.assertTrue(vio_config.enabled)
            self.assertEqual(vio_config.mcap_path, str(mcap))
            kwargs = captured["handpose_kwargs"]
            self.assertIs(kwargs["pose_task"].__class__, _StubTask)
            self.assertEqual(kwargs["world_mcap_path"], world_mcap)
            self.assertEqual(kwargs["qc_path"], str(vio_config.resolved_qc_path))
            self.assertEqual(summary["world_mcap"], world_mcap)

    def test_skip_vio_disables_the_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mcap = Path(tmp) / "raw.mcap"
            mcap.write_bytes(b"placeholder")
            config = main_module.Config(mcap_path=str(mcap), mode="process", skip_vio=True)

            captured: dict = {}

            def fake_start_vio(vio_config: VioConfig):
                captured["vio_config"] = vio_config
                return _StubTask()

            with mock.patch.object(main_module, "start_vio_task", fake_start_vio), mock.patch.object(
                main_module, "readmcap", return_value=(None, [0], 10, 10)
            ), mock.patch.object(
                main_module, "handpose3d", return_value={"head_mcap": "x"}
            ):
                main_module.run_process(config)

            self.assertFalse(captured["vio_config"].enabled)


if __name__ == "__main__":
    unittest.main()
