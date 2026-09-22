# VIO integration (`src/vio/`)

The VIO stage turns a raw recording into a head pose trajectory and fuses it
with the stereo hand keypoints so the output is expressed in the VIO world
frame (origin at the start of the episode, gravity aligned).

## Module map

| module | responsibility |
| --- | --- |
| `vio/rotations.py` | quaternion/rotation primitives (no project deps) |
| `vio/trajectory.py` | TUM loading, interpolation, anchoring, `PoseTrajectory` |
| `vio/frames.py` | body <-> published convention, rigid point transform |
| `vio/mcap_io.py` | camera_info, H.264 frames, IMU, topic discovery |
| `vio/dataset.py` | recording -> Basalt/EuRoC dataset + calibration json |
| `vio/runner.py` | runs the external `basalt_vio` binary |
| `vio/sources.py` | pose sources: Basalt run, TUM file, device `eef_pose` |
| `vio/qc.py` | trajectory coverage/gap metrics and world stability |
| `vio/config.py` | `VioConfig` (JSON `vio` section + CLI overrides) |
| `vio/pipeline.py` | `run_vio`, `PoseTask`, `transform_hand_mcap` |

Only `dataset.py`/`runner.py` know about Basalt; the fusion code consumes the
backend-independent `PoseTrajectory`.

## Local installation

The bootstrap targets Ubuntu 22.04+ on x86_64. Install `uv` following its
[installation guide](https://docs.astral.sh/uv/getting-started/installation/).
On a fresh Ubuntu system, install the system prerequisites once:

```bash
sudo apt-get update
sudo apt-get install curl ca-certificates tar coreutils ffmpeg libegl1 libgl1 libglu1-mesa libx11-6 libxcursor1 libxinerama1 libxrandr2 libxi6 libxtst6
bash scripts/bootstrap.sh
```

The script downloads the [official Basalt 0.1.7 release](https://gitlab.com/VladyslavUsenko/basalt/-/releases/0.1.7),
checks a pinned SHA256, and verifies that `basalt_vio --help` starts successfully
before publishing the installation. Re-running reuses a recognized installation
and checks it again. Existing unrecognized installations are not overwritten.
Use `--basalt-only` to skip `uv sync` and leave the Python environment untouched.

```text
.deps/basalt/0.1.7/bin/basalt_vio     executable
.deps/basalt/0.1.7/lib/              bundled native libraries
.deps/basalt/0.1.7/etc/basalt/        VIO configuration templates
.venv/                              Python environment managed by uv
basalt/                             generated trajectories and datasets
```

The script locates the checkout from its own location, so it can be invoked
from another working directory. Python synchronization always targets this
checkout's `.venv`, even when a different environment is activated. The default
binary and configuration template paths are also derived from the checkout.
Relative `vio.basalt_bin` / `--vio-binary` paths are resolved against the project
root before the runner changes working directory. Library paths are passed
only to the Basalt subprocess; no shell profile edits are needed. An absolute
binary path can still select an external installation; the default configuration
template remains the project-local one.

## Usage

```bash
# full processing run: MediaPipe + Basalt in parallel, world output written
uv run main.py --mode process --mcap-path /path/to/raw.mcap

# without VIO (head-frame output only)
uv run main.py --mode process --mcap-path /path/to/raw.mcap --skip-vio

# standalone VIO stage (kept for compatibility with the old bridge CLI)
uv run src/basalt_bridge.py --mcap_path /path/to/raw.mcap
```

Outputs: `mcap_output/hand_keypoints.mcap` (head frame, unchanged) and
`mcap_output/hand_keypoints_world.mcap` (world frame, same topics plus the 2D
topics), plus `basalt/qc.json` and `basalt/trajectory.txt`.

## Configuration (`configs/ego_config.json`)

```json
"vio": {
  "enabled": true,
  "basalt_bin": "./.deps/basalt/0.1.7/bin/basalt_vio",
  "cam0": 1,
  "cam1": 4,
  "scale": 0.5,
  "anchor": "position",
  "yaw_deg": 89.8,
  "output_dir": "./basalt",
  "keep_frames": false,
  "reuse_trajectory": true
}
```

* `anchor="position"`: origin at the start pose, gravity-aligned axes.
  `anchor="pose"` additionally takes the start orientation.
* `yaw_deg`: rotation about the vertical axis aligning Basalt's world with the
  robot world (measured constant on this rig; re-validate per device).
* `keep_frames=false`: the decoded PNGs (~GBs) are deleted after a successful
  run; the trajectory, calibration and config files stay.
* `reuse_trajectory=true`: a trajectory newer than the recording is reused, so
  re-processing does not re-run Basalt.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

The suite covers the pure math, the config/caching logic and the inline fusion.
`tests/test_vio_regression.py` additionally compares the refactored code with
the pre-refactor implementation when the local artifacts are present.

## Troubleshooting

* **`basalt_vio` not found** – run `bash scripts/bootstrap.sh` or pass `--vio-binary` /
  `basalt_bin`; the pipeline then falls back to the head-frame output and
  records `failed` in the QC report.
* **`only the double sphere ('ds') model is supported`** – the calibration
  expects the EgoScale `ds` camera model; check that `camera_info` comes from
  the same rig generation.
* **world output looks rotated** – check `yaw_deg`; it is device specific.
* **frames outside the trajectory span** – they are clamped to the nearest
  pose; the QC report counts them.
