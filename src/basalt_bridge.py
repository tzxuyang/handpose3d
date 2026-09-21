"""Backwards-compatible CLI for the VIO stage.

The implementation moved into the :mod:`vio` package; this module keeps the
original command line (and its defaults) working:

    uv run src/basalt_bridge.py --mcap_path /path/to/raw.mcap

Everything is delegated to :mod:`vio.pipeline`; see ``vio/`` for the modules.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vio.config import VioConfig  # noqa: E402
from vio.constants import DEFAULT_BASALT_BIN  # noqa: E402
from vio.pipeline import run_pipeline  # noqa: E402


@dataclass
class BasaltRunConfig:
    """Legacy command-line surface (field names and defaults are unchanged)."""

    mcap_path: str = "/home/ubuntu/handpose3D_dataSet/dataset/origin_data_4.mcap"
    output_dir: str = "./basalt"
    ego_config_path: str = "./configs/ego_config.json"
    cam0: int = 1
    cam1: int = 4
    scale: float = 0.5
    max_frames: int = 0
    start_time_s: float = 0.0
    duration_s: float = 0.0
    num_threads: int = 0
    hand_mcap: str = "./mcap_output/hand_keypoints.mcap"
    world_hand_mcap: str = "./mcap_output/hand_keypoints_world.mcap"
    undo_z_rotation: bool = True
    anchor: str = "position"
    yaw_deg: float = 89.8
    skip_export: bool = False
    skip_run: bool = False
    basalt_bin: str = str(DEFAULT_BASALT_BIN)
    keep_marg_data: bool = False
    keep_frames: bool = False
    reuse_trajectory: bool = True


def to_vio_config(config: BasaltRunConfig) -> VioConfig:
    """Convert the legacy CLI dataclass into the package ``VioConfig``."""
    return VioConfig(
        mcap_path=config.mcap_path,
        output_dir=config.output_dir,
        ego_config_path=config.ego_config_path,
        hand_mcap=config.hand_mcap,
        world_hand_mcap=config.world_hand_mcap,
        cam0=config.cam0,
        cam1=config.cam1,
        scale=config.scale,
        max_frames=config.max_frames,
        start_time_s=config.start_time_s,
        duration_s=config.duration_s,
        keep_frames=config.keep_frames,
        basalt_bin=config.basalt_bin,
        num_threads=config.num_threads,
        keep_marg_data=config.keep_marg_data,
        anchor=config.anchor,
        yaw_deg=config.yaw_deg,
        undo_z_rotation=config.undo_z_rotation,
        skip_export=config.skip_export,
        skip_run=config.skip_run,
        reuse_trajectory=config.reuse_trajectory,
    )


def main() -> None:
    import tyro

    config = tyro.cli(BasaltRunConfig)
    print(json.dumps(run_pipeline(to_vio_config(config)), indent=2))


if __name__ == "__main__":
    main()
