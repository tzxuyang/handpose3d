"""``VioConfig``: the single source of truth for the VIO stage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .constants import DEFAULT_BASALT_BIN


@dataclass
class VioConfig:
    """Configuration for export -> VIO -> fuse.

    Defaults match the values that were validated on the DAS Ego recordings;
    ``process`` reads the ``vio`` section of ``configs/ego_config.json`` on top
    of these defaults, and CLI flags override the JSON.
    """

    # input / output
    mcap_path: str = ""
    output_dir: str = "./basalt"
    ego_config_path: str = "./configs/ego_config.json"
    hand_mcap: str = "./mcap_output/hand_keypoints.mcap"
    world_hand_mcap: str = "./mcap_output/hand_keypoints_world.mcap"
    qc_path: str = ""

    # dataset export
    cam0: int = 1
    cam1: int = 4
    scale: float = 0.5
    max_frames: int = 0
    start_time_s: float = 0.0
    duration_s: float = 0.0
    keep_frames: bool = False

    # basalt run
    basalt_bin: str = str(DEFAULT_BASALT_BIN)
    num_threads: int = 0
    keep_marg_data: bool = False
    show_gui: bool = False
    fmt: str = "tum"

    # fusion
    anchor: str = "position"
    yaw_deg: float = 89.8
    undo_z_rotation: bool = True

    # orchestration
    enabled: bool = True
    skip_export: bool = False
    skip_run: bool = False
    reuse_trajectory: bool = True

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VioConfig":
        """Build a config from a JSON-like mapping, ignoring unknown keys."""
        known = {field.name for field in fields(cls)}
        payload = {key: value for key, value in (data or {}).items() if key in known}
        return cls(**payload)

    @classmethod
    def from_ego_config(
        cls,
        config_path: str | os.PathLike = "./configs/ego_config.json",
        **overrides: Any,
    ) -> "VioConfig":
        """Load the ``vio`` section of ``ego_config.json`` plus overrides."""
        section: dict[str, Any] = {}
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as handle:
                payload = json.load(handle)
            section = payload.get("vio") or {}
        config = cls.from_dict(section)
        config.ego_config_path = str(config_path)
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config

    # -- derived paths -----------------------------------------------------
    @property
    def dataset_dir(self) -> Path:
        return Path(self.output_dir) / "dataset"

    @property
    def trajectory_path(self) -> Path:
        name = "trajectory.txt" if self.fmt == "tum" else "trajectory.csv"
        return Path(self.output_dir) / name

    @property
    def resolved_qc_path(self) -> Path:
        return Path(self.qc_path) if self.qc_path else Path(self.output_dir) / "qc.json"

    def validate(self) -> None:
        if not self.mcap_path:
            raise ValueError("VioConfig.mcap_path is required")
        if self.anchor not in ("position", "pose"):
            raise ValueError(f"unknown anchor mode {self.anchor!r}")
        if self.cam0 == self.cam1:
            raise ValueError("cam0 and cam1 must be different cameras")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
