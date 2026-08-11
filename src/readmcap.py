from extract_mcap import extract_video, read_mcap_topics, read_mcap_protobuf, read_mcap_protobuf_once
from utils import write_extrinsic_parameters, write_instrinsic_parameters, quat_to_rot_matrix, msg_time_sync
import json
from matplotlib import pyplot as plt

json_path = "./configs/ego_config.json"
mcap_path = "/home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap"
output_mp4 = "./processed_data/"

# distortion_0 = [-0.013959821820868506, 1.4192899312188747, -0.003533980225025572, -0.0035674091977424003, -6.394324137393883]
# distortion_1 = [-0.02625845479072236, 1.4625979449800532, -0.0014035189373399156, -0.004337224796784845, -6.044993682556749]
distortion = [-0.0, 0.0, 0.0, 0.0, 0.0] # Assuming no distortion for simplicity, replace with actual values if needed

if __name__ == "__main__":
    with open(json_path, "r") as f:
        config = json.load(f)
        video_topics = config.get("camera_topics")
        cam_info = config.get("camera_info")
        imu_topic = config.get("imu_topic")

    # Extract video from MCAP file
    extract_video(mcap_path, video_topics, output_mp4, fps=30, keep_h264=False)

    # Extract camera parameters from MCAP file
    msg_cam0 = read_mcap_protobuf_once(mcap_path, [cam_info[0]])
    msg_cam1 = read_mcap_protobuf_once(mcap_path, [cam_info[1]])

    width, height = msg_cam0.width, msg_cam0.height
    K0 = msg_cam0.K
    K1 = msg_cam1.K
    T_b_c0 = msg_cam0.T_b_c
    T_b_c1 = msg_cam1.T_b_c
    qx_0, qy_0, qz_0, qw_0 = T_b_c0[3], T_b_c0[4], T_b_c0[5], T_b_c0[6]
    qx_1, qy_1, qz_1, qw_1 = T_b_c1[3], T_b_c1[4], T_b_c1[5], T_b_c1[6]
    R0 = quat_to_rot_matrix(qx_0, qy_0, qz_0, qw_0)
    R1 = quat_to_rot_matrix(qx_1, qy_1, qz_1, qw_1)
    T0 = T_b_c0[:3]
    T1 = T_b_c1[:3]
    write_instrinsic_parameters("./camera_parameters/c0.dat", K0, distortion)
    write_instrinsic_parameters("./camera_parameters/c1.dat", K1, distortion)
    write_extrinsic_parameters("./camera_parameters/rot_trans_c0.dat", R0, T0)
    write_extrinsic_parameters("./camera_parameters/rot_trans_c1.dat", R1, T1)

    # Read IMU data
    msg_cam = read_mcap_protobuf(mcap_path, video_topics[0])
    msg_imu = read_mcap_protobuf(mcap_path, imu_topic)
    print(f"IMU data extracted: {len(msg_imu)}")
    print(f"First IMU message: {msg_imu[0]['header']['timestamp'] if msg_imu else 'No IMU data found'}")
    # print("-------------------------------")
    # print(f"First IMU message: {msg_imu[0]['linear_acceleration']['x']}")
    msg_imu_synced = msg_time_sync(msg_cam, msg_imu)
    print(f"IMU data after synchronization: {len(msg_imu_synced)}")

    time_raw = []
    x_accel_raw = []
    time_sync = []
    x_accel_sync = []
    for msg in msg_imu:
        time_raw.append(msg['header']['timestamp'])
        x_accel_raw.append(msg['linear_acceleration']['x'])
    for msg in msg_imu_synced:
        time_sync.append(msg['header']['timestamp'])
        x_accel_sync.append(msg['linear_acceleration']['x'])

    print("Task done")