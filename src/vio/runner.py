"""Running the external ``basalt_vio`` binary.

This is one of the only two modules that know Basalt exists (the other is
``dataset``): everything downstream consumes a ``PoseTrajectory``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .constants import BASALT_INSTALL_HINT, DEFAULT_BASALT_BIN, PROJECT_ROOT


def run_basalt_vio(
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
) -> Path:
    """Run ``basalt_vio`` headless and return the saved trajectory path."""
    binary = Path(binary).expanduser()
    if not binary.is_absolute():
        binary = PROJECT_ROOT / binary
    binary = binary.resolve()
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found; install Basalt with:\n  {BASALT_INSTALL_HINT}")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory = out_dir / ("trajectory.txt" if fmt == "tum" else "trajectory.csv")

    cmd = [
        str(binary),
        "--dataset-path", str(Path(dataset_dir).resolve()),
        "--cam-calib", str(Path(calib_path).resolve()),
        "--dataset-type", "euroc",
        "--config-path", str(Path(config_path).resolve()),
        "--show-gui", "1" if show_gui else "0",
        "--save-trajectory", fmt,
    ]
    if num_threads > 0:
        cmd += ["--num-threads", str(num_threads)]
    if max_frames > 0:
        cmd += ["--max-frames", str(max_frames)]
    if keep_marg_data:
        cmd += ["--marg-data", str(out_dir / "marg_data")]

    env = os.environ.copy()
    lib_dir = binary.parent.parent / "lib"
    if lib_dir.exists():
        previous = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{previous}" if previous else str(lib_dir)

    trajectory.unlink(missing_ok=True)
    if verbose:
        print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(out_dir), env=env, capture_output=True, text=True)
    (out_dir / "basalt_vio.log").write_text(result.stdout + result.stderr)

    if result.returncode != 0 or not trajectory.exists():
        tail = (result.stdout + result.stderr)[-1500:]
        raise RuntimeError(f"basalt_vio failed (exit {result.returncode}); log tail:\n{tail}")
    if verbose:
        print(f"[run] trajectory -> {trajectory}")
    return trajectory
