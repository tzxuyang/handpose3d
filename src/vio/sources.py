"""Pose sources: run Basalt, replay a TUM file, or read the device's eef_pose.

``PoseSource.load()`` always returns a :class:`~vio.trajectory.PoseTrajectory`,
so the fusion code never has to know where the pose came from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from google.protobuf.json_format import MessageToDict

from .constants import DEFAULT_BASALT_BIN
from .rotations import quat_xyzw_to_wxyz
from .runner import run_basalt_vio
from .trajectory import PoseTrajectory, load_tum_trajectory


@runtime_checkable
class PoseSource(Protocol):
    """Anything that can produce a world-frame pose trajectory."""

    def load(self) -> PoseTrajectory:  # pragma: no cover - protocol
        ...


class TumFileSource:
    """Replay a trajectory that was already exported (Basalt TUM format)."""

    def __init__(self, path: str | os.PathLike, **trajectory_kwargs):
        self.path = Path(path)
        self.trajectory_kwargs = trajectory_kwargs

    def load(self) -> PoseTrajectory:
        return PoseTrajectory.from_tum(self.path, **self.trajectory_kwargs)


class BasaltSource:
    """Run ``basalt_vio`` on an exported dataset and load its trajectory."""

    def __init__(
        self,
        dataset_dir: str | os.PathLike,
        calib_path: str | os.PathLike,
        config_path: str | os.PathLike,
        out_dir: str | os.PathLike,
        binary: str | os.PathLike = DEFAULT_BASALT_BIN,
        fmt: str = "tum",
        num_threads: int = 0,
        max_frames: int = 0,
        keep_marg_data: bool = False,
        show_gui: bool = False,
        verbose: bool = True,
        **trajectory_kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.calib_path = calib_path
        self.config_path = config_path
        self.out_dir = out_dir
        self.binary = binary
        self.fmt = fmt
        self.num_threads = num_threads
        self.max_frames = max_frames
        self.keep_marg_data = keep_marg_data
        self.show_gui = show_gui
        self.verbose = verbose
        self.trajectory_kwargs = trajectory_kwargs

    def load(self) -> PoseTrajectory:
        trajectory_path = run_basalt_vio(
            self.dataset_dir,
            self.calib_path,
            self.config_path,
            self.out_dir,
            binary=self.binary,
            fmt=self.fmt,
            num_threads=self.num_threads,
            max_frames=self.max_frames,
            keep_marg_data=self.keep_marg_data,
            show_gui=self.show_gui,
            verbose=self.verbose,
        )
        kwargs = dict(self.trajectory_kwargs)
        kwargs.setdefault("source", f"basalt:{trajectory_path}")
        return PoseTrajectory.from_tum(trajectory_path, **kwargs)


class EefPoseSource:
    """Read the device's own VIO pose topic (``foxglove.PoseInFrame``).

    Recordings that already carry ``/robot0/vio/eef_pose`` can be used to
    validate the offline VIO chain (or to produce world coordinates when no
    Basalt run is required).
    """

    def __init__(
        self,
        mcap_path: str | os.PathLike,
        topic: str = "/robot0/vio/eef_pose",
        **trajectory_kwargs,
    ):
        self.mcap_path = mcap_path
        self.topic = topic
        self.trajectory_kwargs = trajectory_kwargs

    def load(self) -> PoseTrajectory:
        times, quats, positions = [], [], []
        with open(self.mcap_path, "rb") as handle:
            reader = make_reader(handle, decoder_factories=[DecoderFactory()])
            for _schema, _channel, message, proto in reader.iter_decoded_messages(
                topics=[self.topic]
            ):
                if proto is None:
                    continue
                data = MessageToDict(proto, preserving_proto_field_name=True)
                header = data.get("header", {}) or {}
                raw_timestamp = header.get("timestamp")
                timestamp_ns = int(raw_timestamp) if raw_timestamp else int(message.log_time)
                pose = data.get("pose", {}) or {}
                position = pose.get("position", {}) or {}
                orientation = pose.get("orientation", {}) or {}
                times.append(timestamp_ns)
                positions.append(
                    [
                        float(position.get("x", 0.0)),
                        float(position.get("y", 0.0)),
                        float(position.get("z", 0.0)),
                    ]
                )
                quats.append(
                    quat_xyzw_to_wxyz(
                        [
                            float(orientation.get("x", 0.0)),
                            float(orientation.get("y", 0.0)),
                            float(orientation.get("z", 0.0)),
                            float(orientation.get("w", 1.0)),
                        ]
                    )
                )
        if not times:
            raise RuntimeError(f"no poses on {self.topic} in {self.mcap_path}")
        kwargs = dict(self.trajectory_kwargs)
        kwargs.setdefault("source", f"eef_pose:{self.topic}")
        return PoseTrajectory(
            np.asarray(times, dtype=np.int64),
            np.asarray(quats, dtype=float),
            np.asarray(positions, dtype=float),
            **kwargs,
        )
