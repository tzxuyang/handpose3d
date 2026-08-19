import os
import sys
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
sys.path.insert(0, os.path.join(project_root, 'src'))

from imu_calculation import calculate_position_from_imu
from src.mcap_utils import read_hand_json_2d, read_hand_json_3d, read_hand_message_2d, read_mcap_json, read_mcap_protobuf, readmcap, read_hand_message_3d
from src.handpose3d import handpose3d
from src.show_hands import visualize_2d, visualize_3d

config_path = "./configs/ego_config.json"
mcap_path = "/home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap"
mcap_path = "/home/yang/Downloads/0fa832d8c9814bd98ca33461f24a8bbf.mcap"
# mcap_path = '/Users/yangxu/Documents/Code/ff9e3e1189504041b9ce21256925377f.mcap'
output_mp4 = "./processed_data/"
output_mcap_path = "./mcap_output/hand_keypoints.mcap"

mode = "visualize"  # "process" or "visualize"
selected_2d_channel = 1

if __name__ == "__main__":
    if mode == "process":
        with open(config_path, "r") as f:
            config = json.load(f)
            video_topics = config.get("camera_topics")
            cam_info = config.get("camera_info")
            imu_topic = config.get("imu_topic")

        # read the input mcap file and extract camera videos
        msg_imu_synced, timestamps, height, width = readmcap(mcap_path, config_path, output_mp4)

        # detect hand keypoints in 2D and 3D using mediapipe for each camera frame
        imu_pts = calculate_position_from_imu(msg_imu_synced)
        input_streams = [f'processed_data/camera{i}.mp4' for i in range(6)]
        handpose3d(input_streams, output_mcap_path, cam_3d_ids=[1, 4], imu_pts=imu_pts, timestamps=timestamps, visualize=True)
    elif mode == "visualize":
        with open(config_path, "r") as f:
            config = json.load(f)
            video_topics = config.get("camera_topics")
            handpoints_3d_topic = config.get("handpoints_3d_topic")
            handpoints_2d_topic = config.get("handpoints_2d_topic")

        # read the input mcap file of camera videos and visualize the 2D hand keypoints
        video_stream_0 = f'processed_data/camera1.mp4'
        video_stream_1 = f'processed_data/camera4.mp4'
        print(handpoints_2d_topic[1])
        msg_2d_0 = read_mcap_json(output_mcap_path, handpoints_2d_topic[1])
        msg_2d_1 = read_mcap_json(output_mcap_path, handpoints_2d_topic[4])

        p2ds_0 = []
        for frame in msg_2d_0:
            kpts = read_hand_json_2d(frame)
            p2ds_0.append(kpts)
        p2ds_1 = []
        for frame in msg_2d_1:
            kpts = read_hand_json_2d(frame)
            p2ds_1.append(kpts)
        visualize_2d(video_stream_0, video_stream_1, p2ds_0, p2ds_1)

        # read the input mcap file of 3d points and visualize the 3D hand keypoints
        msg_left = read_mcap_json(output_mcap_path, handpoints_3d_topic[0])
        msg_right = read_mcap_json(output_mcap_path, handpoints_3d_topic[1])

        p3ds = []
        for frame_left, frame_right in zip(msg_left, msg_right):
            left_kpts = read_hand_json_3d(frame_left)
            right_kpts = read_hand_json_3d(frame_right)
            print(f"left_kpts: {left_kpts}, right_kpts: {right_kpts}")
            p3ds.append([left_kpts, right_kpts])

        visualize_3d(p3ds)
