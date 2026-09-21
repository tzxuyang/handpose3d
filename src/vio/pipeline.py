"""Orchestration: export -> Basalt -> fuse, with caching and error handling."""

from __future__ import annotations

import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import VioConfig
from .dataset import export_euroc_dataset
from .mcap_io import load_ego_config
from .qc import trajectory_qc, write_qc_json
from .runner import run_basalt_vio
from .trajectory import PoseTrajectory
from ._deps import read_hand_json_3d, read_mcap_json, write_3d_hand_keypoints_mcap

HAND_TOPICS = ("/robot0/handtracking/left", "/robot0/handtracking/right")


@dataclass
class VioResult:
    """Outcome of the VIO stage; ``status`` is ``ok``/``skipped``/``failed``."""

    status: str
    message: str = ""
    trajectory_path: Optional[str] = None
    trajectory: Optional[PoseTrajectory] = None
    qc: dict = field(default_factory=dict)
    export_info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.trajectory is not None


class PoseTask:
    """Asynchronous handle for one VIO run (used by the inline process path)."""

    def __init__(self, future: Future, executor: ThreadPoolExecutor):
        self._future = future
        self._executor = executor

    def done(self) -> bool:
        return self._future.done()

    def get(self, timeout: Optional[float] = None) -> VioResult:
        """Block until the VIO stage finished and return its result."""
        try:
            result = self._future.result(timeout=timeout)
        except Exception as exc:  # defensive: run_vio normally catches its errors
            result = VioResult(status="failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            self._executor.shutdown(wait=False)
        return result


def start_vio_task(config: VioConfig) -> PoseTask:
    """Start the VIO stage in a background thread and return a handle."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vio")
    future = executor.submit(run_vio, config)
    return PoseTask(future, executor)


def _trajectory_is_fresh(config: VioConfig) -> bool:
    trajectory = config.trajectory_path
    source = Path(config.mcap_path)
    if not trajectory.exists() or not source.exists():
        return False
    return trajectory.stat().st_mtime >= source.stat().st_mtime


def _clean_frames(config: VioConfig) -> None:
    """Drop the decoded PNGs; the trajectory and calibration files stay."""
    data_dirs = (
        config.dataset_dir / "mav0" / "cam0" / "data",
        config.dataset_dir / "mav0" / "cam1" / "data",
    )
    for directory in data_dirs:
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


def run_vio(config: VioConfig) -> VioResult:
    """Run export + Basalt (or reuse a cached trajectory) and return the poses."""
    if not config.enabled:
        return VioResult(status="skipped", message="VIO disabled by configuration")

    try:
        config.validate()
    except ValueError as exc:
        return VioResult(status="failed", message=str(exc))

    export_info: dict = {}
    trajectory_path = config.trajectory_path

    use_cache = config.reuse_trajectory and _trajectory_is_fresh(config)
    if use_cache:
        if not config.keep_marg_data:
            pass  # nothing to clean when we did not run
        try:
            trajectory = PoseTrajectory.from_tum(
                trajectory_path,
                anchor=config.anchor,
                yaw_deg=config.yaw_deg,
                source=f"basalt-cache:{trajectory_path}",
            )
        except Exception as exc:
            return VioResult(status="failed", message=f"cached trajectory unusable: {exc}")
        qc = trajectory_qc(trajectory)
        qc["cached"] = True
        return VioResult(
            status="ok",
            message=f"reused {trajectory_path}",
            trajectory_path=str(trajectory_path),
            trajectory=trajectory,
            qc=qc,
        )

    try:
        if not config.skip_export:
            export_info = export_euroc_dataset(
                config.mcap_path,
                config.dataset_dir,
                config.ego_config_path,
                cam_ids=(config.cam0, config.cam1),
                scale=config.scale,
                max_frames=config.max_frames,
                start_time_s=config.start_time_s,
                duration_s=config.duration_s,
            )
        else:
            export_info = {
                "dataset_dir": str(config.dataset_dir),
                "calib_path": str(config.dataset_dir / "basalt_calib.json"),
                "config_path": str(config.dataset_dir / "basalt_config.json"),
            }

        if config.skip_run:
            if not trajectory_path.exists():
                raise FileNotFoundError(
                    f"{trajectory_path} not found; run without skip_run first"
                )
        else:
            trajectory_path = run_basalt_vio(
                export_info["dataset_dir"],
                export_info["calib_path"],
                export_info["config_path"],
                config.output_dir,
                binary=config.basalt_bin,
                fmt=config.fmt,
                num_threads=config.num_threads,
                max_frames=config.max_frames,
                keep_marg_data=config.keep_marg_data,
                show_gui=config.show_gui,
            )

        trajectory = PoseTrajectory.from_tum(
            trajectory_path,
            anchor=config.anchor,
            yaw_deg=config.yaw_deg,
            source=f"basalt:{trajectory_path}",
        )
        qc = trajectory_qc(trajectory, export_info.get("frame_timestamps_ns"))
        qc["cached"] = False
        write_qc_json(config.resolved_qc_path, qc)

        if not config.keep_frames:
            _clean_frames(config)

        return VioResult(
            status="ok",
            message=f"ran Basalt on {export_info.get('frames', '?')} frames",
            trajectory_path=str(trajectory_path),
            trajectory=trajectory,
            qc=qc,
            export_info=export_info,
        )
    except Exception as exc:
        return VioResult(status="failed", message=f"{type(exc).__name__}: {exc}")


def transform_hand_mcap(
    input_mcap: str,
    trajectory: PoseTrajectory | str,
    output_mcap: str,
    *,
    undo_z_rotation: bool = True,
    t_imu_body: np.ndarray | None = None,
    anchor: str = "position",
    yaw_deg: float = 89.8,
    verbose: bool = True,
) -> str:
    """Rewrite a 3D hand keypoint mcap in the VIO world frame.

    ``trajectory`` may be a :class:`PoseTrajectory` or a path to a TUM file;
    the latter is turned into a trajectory with ``anchor``/``yaw_deg``.
    """
    if isinstance(trajectory, (str, Path)):
        trajectory = PoseTrajectory.from_tum(trajectory, anchor=anchor, yaw_deg=yaw_deg)

    frames = []
    timestamps = None
    for index, topic in enumerate(HAND_TOPICS):
        messages = read_mcap_json(str(input_mcap), topic)
        if not messages:
            raise RuntimeError(f"no messages on {topic} in {input_mcap}")
        frames.append(
            np.asarray([read_hand_json_3d(message) for message in messages], dtype=float)
        )
        if index == 0:
            timestamps = [int(message["log_time"]) for message in messages]

    frame_count = min(len(timestamps), frames[0].shape[0], frames[1].shape[0])
    timestamps = np.asarray(timestamps[:frame_count], dtype=np.int64)
    hands = np.stack([frames[0][:frame_count], frames[1][:frame_count]], axis=1)

    world = trajectory.transform_points(
        hands, timestamps, undo_published_rotation=undo_z_rotation
    )

    if verbose:
        outside = trajectory.outside_span(timestamps)
        print(
            f"[apply] {frame_count} frames, {int((~outside).sum())} inside the VIO time span "
            f"({trajectory.duration_s:.1f}s)"
        )
        if abs(trajectory.yaw_deg) > 1e-9:
            print(f"[apply] yaw {trajectory.yaw_deg:+.2f} deg about the vertical (gravity) axis")
        print(f"[apply] anchor mode: {trajectory.anchor}")
        if outside.any():
            print("  ! frames outside the trajectory span are clamped to the nearest pose")

    output_path = Path(output_mcap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_3d_hand_keypoints_mcap(world, [int(value) for value in timestamps], str(output_path))
    if verbose:
        print(f"[apply] world-frame hand keypoints -> {output_path}")
    return str(output_path)


def run_pipeline(config: VioConfig) -> dict:
    """Legacy entry point: export, run VIO and transform an existing hand mcap."""
    result = run_vio(config)
    if not result.ok:
        raise RuntimeError(f"VIO stage {result.status}: {result.message}")

    world_mcap = None
    if config.hand_mcap and Path(config.hand_mcap).exists():
        world_mcap = transform_hand_mcap(
            config.hand_mcap,
            result.trajectory,
            config.world_hand_mcap,
            undo_z_rotation=config.undo_z_rotation,
        )
    else:
        print(f"[apply] skipped: {config.hand_mcap} not found (run --mode process first)")

    return {
        "dataset_dir": result.export_info.get("dataset_dir", str(config.dataset_dir)),
        "trajectory": result.trajectory_path,
        "world_hand_mcap": world_mcap,
        "qc": result.qc,
    }
