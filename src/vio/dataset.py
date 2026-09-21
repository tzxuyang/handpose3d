"""Recording -> Basalt/EuRoC dataset conversion and calibration writing."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .constants import DEFAULT_BASALT_ETC, STD_GRAVITY
from .mcap_io import (
    decode_h264_sequence,
    load_ego_config,
    read_camera_info,
    read_camera_messages,
    read_imu_samples,
    write_csv,
)
from .rotations import se3_row_to_pose


def write_basalt_calib(
    cam_infos: list[dict],
    out_path: str | os.PathLike,
    scale: float = 1.0,
    imu_rate: float = 200.0,
    cam_time_offset_ns: int = 0,
) -> str:
    """Write a Basalt calibration json from EgoScale ``camera_info`` messages.

    ``camera_info.T_b_c`` is written as ``T_imu_cam``: it maps the rig body
    frame to the camera frame and the body frame is treated as the IMU frame
    (both sit at the optical-rig origin to within a few centimetres).
    """
    if len(cam_infos) != 2:
        raise ValueError("Basalt's euroc reader expects exactly two cameras")
    for info in cam_infos:
        if info["distortion_model"] != "ds":
            raise ValueError(
                f"only the double sphere ('ds') model is supported, got {info['distortion_model']!r}"
            )

    t_imu_cam, intrinsics, resolutions = [], [], []
    for info in cam_infos:
        fx, fy, cx, cy, xi, alpha = info["D"][:6]
        intrinsics.append(
            {
                "camera_type": "ds",
                "intrinsics": {
                    "fx": fx * scale,
                    "fy": fy * scale,
                    "cx": cx * scale,
                    "cy": cy * scale,
                    "xi": xi,
                    "alpha": alpha,
                },
            }
        )
        resolutions.append([int(round(info["width"] * scale)), int(round(info["height"] * scale))])

        position, quat_wxyz = se3_row_to_pose(info["T_b_c"])
        t_imu_cam.append(
            {
                "px": float(position[0]),
                "py": float(position[1]),
                "pz": float(position[2]),
                "qx": float(quat_wxyz[1]),
                "qy": float(quat_wxyz[2]),
                "qz": float(quat_wxyz[3]),
                "qw": float(quat_wxyz[0]),
            }
        )

    payload = {
        "value0": {
            "T_imu_cam": t_imu_cam,
            "intrinsics": intrinsics,
            "resolution": resolutions,
            # all-ones radial attenuation = no vignetting correction
            "vignette": [
                {"value0": 0, "value1": 50000000000, "value2": [[1.0]] * 15},
                {"value0": 0, "value1": 50000000000, "value2": [[1.0]] * 15},
            ],
            "calib_accel_bias": [0.0] * 9,
            "calib_gyro_bias": [0.0] * 12,
            "imu_update_rate": imu_rate,
            "accel_noise_std": [0.016, 0.016, 0.016],
            "gyro_noise_std": [0.000282, 0.000282, 0.000282],
            "accel_bias_std": [0.001, 0.001, 0.001],
            "gyro_bias_std": [0.0001, 0.0001, 0.0001],
            "T_mocap_world": {
                "px": 0.0, "py": 0.0, "pz": 0.0,
                "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            },
            "T_imu_marker": {
                "px": 0.0, "py": 0.0, "pz": 0.0,
                "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            },
            "mocap_time_offset_ns": 0,
            "mocap_to_imu_offset_ns": 0,
            "cam_time_offset_ns": int(cam_time_offset_ns),
        }
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return str(out_path)


def write_basalt_config(
    out_path: str | os.PathLike,
    scale: float = 1.0,
    template: Path = DEFAULT_BASALT_ETC / "euroc_config.json",
) -> str:
    """Write the Basalt VIO config, based on the installed template when present."""
    if template.exists():
        with open(template, "r") as handle:
            payload = json.load(handle)
    else:
        payload = {
            "value0": {
                "config.optical_flow_type": "frame_to_frame",
                "config.optical_flow_detection_grid_size": 50,
                "config.optical_flow_levels": 3,
                "config.vio_linearization_type": "ABS_QR",
                "config.vio_sqrt_marg": True,
                "config.vio_max_states": 3,
                "config.vio_max_kfs": 7,
                "config.vio_min_frames_after_kf": 5,
                "config.vio_obs_std_dev": 0.5,
                "config.vio_enforce_realtime": False,
            }
        }

    values = payload.setdefault("value0", {})
    grid_key = "config.optical_flow_detection_grid_size"
    if abs(scale - 1.0) > 1e-9 and grid_key in values:
        # keep the physical feature density when the images are downscaled
        values[grid_key] = max(8, int(round(float(values[grid_key]) * scale)))
    values.setdefault("config.vio_enforce_realtime", False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return str(out_path)


def export_euroc_dataset(
    mcap_path: str,
    out_dir: str | os.PathLike,
    ego_config_path: str | os.PathLike = "./configs/ego_config.json",
    cam_ids: tuple[int, int] = (1, 4),
    scale: float = 0.5,
    max_frames: int = 0,
    start_time_s: float = 0.0,
    duration_s: float = 0.0,
    gray: bool = True,
    imu_scale: float = STD_GRAVITY,
    verbose: bool = True,
) -> dict:
    """Write ``<out_dir>/mav0`` (EuRoC layout) plus calibration and config json.

    ``cam_ids`` must be two cameras of the same hardware sync group; the rig
    uses ``{0, 1, 4, 5}`` and ``{2, 3}``, so the default ``(1, 4)`` pair - the
    cameras that already triangulate the hands - is a valid stereo pair.
    """
    config = load_ego_config(ego_config_path)
    camera_topics = config["camera_topics"]
    imu_topic = config["imu_topic"]
    cam0_topic, cam1_topic = camera_topics[cam_ids[0]], camera_topics[cam_ids[1]]

    out_dir = Path(out_dir)
    mav0 = out_dir / "mav0"
    cam0_dir = mav0 / "cam0" / "data"
    cam1_dir = mav0 / "cam1" / "data"
    for directory in (cam0_dir, cam1_dir, mav0 / "imu0"):
        directory.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[export] reading {cam0_topic} and {cam1_topic}")
    times0, payloads0 = read_camera_messages(mcap_path, cam0_topic)
    times1, payloads1 = read_camera_messages(mcap_path, cam1_topic)

    pair_count = min(len(times0), len(payloads0), len(times1), len(payloads1))
    if len(times0) != len(times1) and verbose:
        print(f"  ! camera frame counts differ: {len(times0)} vs {len(times1)}; using {pair_count}")

    first_ns = times0[0] + int(start_time_s * 1e9)
    last_ns = first_ns + int(duration_s * 1e9) if duration_s > 0 else times0[pair_count - 1]
    selected = [index for index in range(pair_count) if first_ns <= times0[index] <= last_ns]
    if max_frames > 0:
        selected = selected[:max_frames]
    if len(selected) < 10:
        raise RuntimeError("not enough frames selected for VIO (need at least 10)")

    times0 = [times0[index] for index in selected]
    times1 = [times1[index] for index in selected]
    payloads0 = [payloads0[index] for index in selected]
    payloads1 = [payloads1[index] for index in selected]

    skew = (max(times1[0] - times0[0], 0), max(abs(times1[-1] - times0[-1]), 0))
    if max(skew) > 5e6 and verbose:
        print(f"  ! the two cameras look badly synchronised ({max(skew) / 1e6:.1f} ms apart)")

    if verbose:
        print(f"[export] decoding {len(selected)} stereo frames (scale={scale})")
    for directory in (cam0_dir, cam1_dir):  # drop stale frames from earlier runs
        for stale in list(directory.glob("*_*.png")):
            stale.unlink()
    frames0, shift0 = decode_h264_sequence(payloads0, cam0_dir, "cam0", scale, gray, verbose=verbose)
    frames1, shift1 = decode_h264_sequence(payloads1, cam1_dir, "cam1", scale, gray, verbose=verbose)
    if shift0 != shift1 and verbose:
        print(f"  ! camera decode offsets differ ({shift0} vs {shift1}); check the export")

    frame_count = min(len(frames0), len(frames1), len(times0) - max(shift0, shift1))
    if frame_count < 10:
        raise RuntimeError("fewer than 10 decoded stereo frames")

    # frames[i] belongs to message i + shift, so start the timestamps at `shift`
    frames0 = frames0[:frame_count]
    frames1 = frames1[:frame_count]
    times0 = times0[shift0 : shift0 + frame_count]

    # Basalt derives the file name of *both* cameras from cam0's data.csv, so the
    # two image folders must contain identically named files.
    rows: list[list[object]] = []
    for index in range(frame_count):
        name = f"{times0[index]}.png"
        rows.append([times0[index], name])
        frames0[index].replace(cam0_dir / name)
        frames1[index].replace(cam1_dir / name)
    write_csv(mav0 / "cam0" / "data.csv", rows)
    write_csv(mav0 / "cam1" / "data.csv", rows)

    imu_times, imu_gyro, imu_accel = read_imu_samples(mcap_path, imu_topic)
    margin_ns = int(0.2e9)
    window = (imu_times >= times0[0] - margin_ns) & (imu_times <= times0[-1] + margin_ns)
    imu_rows = [
        [int(time_ns)]
        + [f"{value:.9f}" for value in gyro]
        + [f"{value * imu_scale:.9f}" for value in accel]
        for time_ns, gyro, accel in zip(imu_times[window], imu_gyro[window], imu_accel[window])
    ]
    write_csv(mav0 / "imu0" / "data.csv", imu_rows)

    cam_infos = [
        read_camera_info(mcap_path, config["camera_info"][0]),
        read_camera_info(mcap_path, config["camera_info"][1]),
    ]
    calib_path = write_basalt_calib(cam_infos, out_dir / "basalt_calib.json", scale=scale, imu_rate=200.0)
    config_path = write_basalt_config(out_dir / "basalt_config.json", scale=scale)

    duration = (times0[-1] - times0[0]) * 1e-9
    if verbose:
        print(
            f"[export] {frame_count} stereo frames, {int(window.sum())} imu samples, "
            f"{duration:.1f}s, cameras {cam_infos[0]['width']}x{cam_infos[0]['height']} "
            f"-> {scale:.2f}x -> {out_dir}"
        )
    return {
        "dataset_dir": str(out_dir),
        "calib_path": calib_path,
        "config_path": config_path,
        "camera_topics": [cam0_topic, cam1_topic],
        "frames": frame_count,
        "duration_s": duration,
        "first_timestamp_ns": times0[0],
        "last_timestamp_ns": times0[-1],
        "frame_timestamps_ns": times0,
    }
