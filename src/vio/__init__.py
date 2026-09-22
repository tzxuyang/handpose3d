"""VIO integration package for the handpose3d pipeline.

The package is split by responsibility:

``rotations``   quaternion / rotation primitives (no project dependencies)
``trajectory``  TUM trajectories, interpolation, anchoring, ``PoseTrajectory``
``frames``      body <-> published frame conventions and point transforms
``mcap_io``     reading recordings (camera_info, frames, IMU)
``dataset``     recording -> Basalt/EuRoC dataset + calibration files
``runner``      running the external ``basalt_vio`` binary
``sources``     pose sources (Basalt run, TUM file, device eef_pose)
``qc``          quality metrics for the fused trajectory
``config``      ``VioConfig`` dataclass and its JSON/CLI plumbing
``pipeline``    orchestration: export -> run -> fuse, plus the task handle

Only ``dataset``/``runner`` know about Basalt; everything else consumes the
neutral ``PoseTrajectory`` interface, so another backend can be dropped in.
"""

from .config import VioConfig
from .pipeline import PoseTask, VioResult, run_vio, start_vio_task
from .trajectory import PoseTrajectory, load_tum_trajectory

__all__ = [
    "PoseTask",
    "PoseTrajectory",
    "VioConfig",
    "VioResult",
    "load_tum_trajectory",
    "run_vio",
    "start_vio_task",
]
