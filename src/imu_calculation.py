import copy

import numpy as np


def _get_linear_acceleration(message):
    if "linear_acceleration" in message:
        return message["linear_acceleration"]
    if "linear_acceleraion" in message:
        return message["linear_acceleraion"]
    raise KeyError("IMU message must contain 'linear_acceleration' (or legacy 'linear_acceleraion').")


def _parse_timestamp(timestamp):
    if isinstance(timestamp, dict):
        seconds = float(timestamp.get("sec", 0.0))
        nanoseconds = float(timestamp.get("nanosec", 0.0))
        return seconds + nanoseconds * 1e-9
    return float(timestamp)


def _normalize_timestamps(timestamps):
    timestamp_array = np.asarray([_parse_timestamp(ts) for ts in timestamps], dtype=float)
    if timestamp_array.size < 2:
        return timestamp_array

    dt = np.diff(timestamp_array)
    positive_dt = dt[dt > 0]
    if positive_dt.size == 0:
        return timestamp_array

    median_dt = np.median(positive_dt)
    if median_dt > 1e6:
        return timestamp_array * 1e-9
    if median_dt > 1e3:
        return timestamp_array * 1e-6
    if median_dt > 1:
        return timestamp_array * 1e-3
    return timestamp_array


def _extract_imu_arrays(imu_data, timestamps=None):
    if isinstance(imu_data, np.ndarray):
        imu_array = np.asarray(imu_data, dtype=float)
        if imu_array.ndim != 2 or imu_array.shape[1] != 6:
            raise ValueError("imu_data must have shape (N, 6): [gx, gy, gz, ax, ay, az].")
        gyro = imu_array[:, :3]
        accel = imu_array[:, 3:]
        timestamp_array = None if timestamps is None else _normalize_timestamps(timestamps)
        return gyro, accel, timestamp_array, "array"

    messages = list(imu_data)
    gyro = []
    accel = []
    extracted_timestamps = [] if timestamps is None else timestamps

    for message in messages:
        linear_acceleration = _get_linear_acceleration(message)
        gyro.append([
            float(message["angular_velocity"]["x"]),
            float(message["angular_velocity"]["y"]),
            float(message["angular_velocity"]["z"]),
        ])
        accel.append([
            float(linear_acceleration["x"]),
            float(linear_acceleration["y"]),
            float(linear_acceleration["z"]),
        ])
        if timestamps is None:
            extracted_timestamps.append(message["header"]["timestamp"])

    timestamp_array = None
    if extracted_timestamps is not None:
        timestamp_array = _normalize_timestamps(extracted_timestamps)

    return np.asarray(gyro, dtype=float), np.asarray(accel, dtype=float), timestamp_array, "messages"


def _compute_dt(sample_count, timestamps=None, sample_rate=None):
    if sample_count == 0:
        return np.zeros(0, dtype=float)

    if timestamps is not None:
        if len(timestamps) != sample_count:
            raise ValueError("timestamps must have the same length as imu_data.")
        dt = np.diff(timestamps)
        if np.any(dt < 0):
            raise ValueError("timestamps must be non-decreasing.")
        return dt

    if sample_rate is None:
        sample_rate = 1.0
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    return np.full(sample_count - 1, 1.0 / float(sample_rate), dtype=float)


def _compute_sample_times(sample_count, timestamps=None, sample_rate=None):
    if sample_count == 0:
        return np.zeros(0, dtype=float)

    if timestamps is not None:
        sample_times = np.asarray(timestamps, dtype=float)
    else:
        if sample_rate is None:
            sample_rate = 1.0
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        sample_times = np.arange(sample_count, dtype=float) / float(sample_rate)

    return sample_times - sample_times[0]


def _subtract_moving_average(signal, sample_times, window_seconds):
    if window_seconds <= 0:
        raise ValueError("bias_window_seconds must be positive.")
    if len(signal) == 0:
        return signal.copy()

    corrected_signal = np.zeros_like(signal)
    prefix_sum = np.vstack([np.zeros((1, signal.shape[1])), np.cumsum(signal, axis=0)])
    start_index = 0

    for index in range(len(signal)):
        cutoff_time = sample_times[index] - window_seconds
        while start_index < index and sample_times[start_index] < cutoff_time:
            start_index += 1

        window_count = index - start_index + 1
        window_sum = prefix_sum[index + 1] - prefix_sum[start_index]
        corrected_signal[index] = signal[index] - window_sum / window_count

    return corrected_signal


def _skew_symmetric(vector):
    return np.array([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])


def _rotation_from_angular_step(angular_step):
    angle = np.linalg.norm(angular_step)
    if angle < 1e-12:
        return np.eye(3) + _skew_symmetric(angular_step)

    axis = angular_step / angle
    skew_axis = _skew_symmetric(axis)
    return (
        np.eye(3)
        + np.sin(angle) * skew_axis
        + (1.0 - np.cos(angle)) * (skew_axis @ skew_axis)
    )


def _integrate_sensor_rotations(gyro, dt):
    rotations = np.repeat(np.eye(3)[None, :, :], len(gyro), axis=0)
    for index in range(1, len(gyro)):
        delta_rotation = _rotation_from_angular_step(gyro[index - 1] * dt[index - 1])
        rotations[index] = rotations[index - 1] @ delta_rotation
    return rotations


def _gravity_compensated_acceleration(gyro, accel, timestamps=None, sample_rate=None, gravity_magnitude=1.0):
    dt = _compute_dt(len(accel), timestamps=timestamps, sample_rate=sample_rate)
    rotations = _integrate_sensor_rotations(gyro, dt)
    gravity_world = np.array([0.0, 0.0, gravity_magnitude], dtype=float)
    gravity_sensor = np.einsum("nji,j->ni", rotations, gravity_world)
    linear_accel_sensor = accel - gravity_sensor
    return linear_accel_sensor, rotations


def subtract_gravity_from_imu(imu_data, timestamps=None, sample_rate=None, gravity_magnitude=1.0):
    """
    Remove gravity from IMU samples.

    Supports IMU messages with:
    - message["header"]["timestamp"]
    - message["angular_velocity"]["x"|"y"|"z"]
    - message["linear_acceleration"]["x"|"y"|"z"]

    Assumes the IMU starts with its +z axis aligned with gravity and integrates
    the angular velocity to keep track of the gravity direction in sensor frame.
    """
    imu_source = imu_data if isinstance(imu_data, np.ndarray) else list(imu_data)
    gyro, accel, extracted_timestamps, input_kind = _extract_imu_arrays(imu_source, timestamps=timestamps)
    active_timestamps = extracted_timestamps if timestamps is None else _normalize_timestamps(timestamps)
    linear_accel_sensor, _ = _gravity_compensated_acceleration(
        gyro,
        accel,
        timestamps=active_timestamps,
        sample_rate=sample_rate,
        gravity_magnitude=gravity_magnitude,
    )

    if input_kind == "array":
        return np.concatenate([gyro, linear_accel_sensor], axis=1)

    output_messages = copy.deepcopy(imu_source)
    for message, corrected_accel in zip(output_messages, linear_accel_sensor):
        linear_acceleration = _get_linear_acceleration(message)
        linear_acceleration["x"] = float(corrected_accel[0])
        linear_acceleration["y"] = float(corrected_accel[1])
        linear_acceleration["z"] = float(corrected_accel[2])
    return output_messages


def calculate_position_from_imu(
    imu_data,
    timestamps=None,
    sample_rate=None,
    gravity_magnitude=1.0,
    bias_window_seconds=3.5,
):
    """
    Estimate a relative (x, y, z) trajectory from IMU data.

    Supports the IMU message dictionary format used by this repository.

    Gravity is removed first, the remaining acceleration is rotated into the
    world frame, then moving-average drift compensation is applied to both the
    integrated velocity and the integrated position over a past-time window.
    The returned trajectory always starts at (0, 0, 0).
    """
    gyro, accel, extracted_timestamps, _ = _extract_imu_arrays(imu_data, timestamps=timestamps)
    active_timestamps = extracted_timestamps if timestamps is None else _normalize_timestamps(timestamps)
    linear_accel_sensor, rotations = _gravity_compensated_acceleration(
        gyro,
        accel,
        timestamps=active_timestamps,
        sample_rate=sample_rate,
        gravity_magnitude=gravity_magnitude,
    )

    linear_accel_world = np.einsum("nij,nj->ni", rotations, linear_accel_sensor)
    if len(linear_accel_world) == 0:
        return linear_accel_world

    dt = _compute_dt(len(linear_accel_world), timestamps=active_timestamps, sample_rate=sample_rate)
    sample_times = _compute_sample_times(
        len(linear_accel_world),
        timestamps=active_timestamps,
        sample_rate=sample_rate,
    )

    velocity = np.zeros_like(linear_accel_world)
    true_accel_world = np.zeros_like(linear_accel_world)

    for index in range(1, len(linear_accel_world)):
        bias = np.mean(linear_accel_world[max(0, index - int(bias_window_seconds / dt[index - 1])):index], axis=0)
        true_accel_world[index] = linear_accel_world[index] - bias

    for index in range(1, len(linear_accel_world)): 
        step = dt[index - 1]
        velocity[index] = velocity[index - 1] + 0.5 * (
            true_accel_world[index - 1] + true_accel_world[index]
        ) * step

    position = np.zeros_like(linear_accel_world)
    true_velocity = np.zeros_like(linear_accel_world)
    for index in range(1, len(velocity)):
        bias = np.mean(velocity[max(0, index - int(bias_window_seconds / dt[index - 1])):index], axis=0)
        true_velocity[index] = velocity[index] - bias

    for index in range(1, len(velocity)):
        step = dt[index - 1]
        position[index] = position[index - 1] + 0.5 * (
            true_velocity[index - 1] + true_velocity[index]
        ) * step

    # position = position - position[0]
    return position

if __name__ == "__main__":
    import json
    from matplotlib import pyplot as plt
    from readmcap import read_mcap_protobuf
    from utils import msg_time_sync

    mcap_path = "/home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap"
    json_path = "./configs/ego_config.json"

    with open(json_path, "r") as f:
        config = json.load(f)
        imu_topic = config.get("imu_topic")
        video_topics = config.get("camera_topics")

    msg_imu = read_mcap_protobuf(mcap_path, imu_topic)
    msg_cam = read_mcap_protobuf(mcap_path, video_topics[0])
    msg_imu_synced = msg_time_sync(msg_cam, msg_imu)
    position = calculate_position_from_imu(msg_imu_synced)
    print(f"Estimated position shape: {position.shape}")
    x_list = []
    y_list = []
    z_list = []
    for pos in position:
        x_list.append(pos[0])
        y_list.append(pos[1])
        z_list.append(pos[2])

    plt.figure(figsize=(12, 6))
    plt.plot(x_list, label='X Position', color='blue')
    plt.plot(y_list, label='Y Position', color='orange')
    plt.plot(z_list, label='Z Position', color='green')
    plt.xlabel('Sample Index')
    plt.ylabel('Position')
    plt.title('Estimated Position from IMU Data')
    plt.legend()
    plt.show()

    