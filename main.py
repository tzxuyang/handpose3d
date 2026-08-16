import os
import sys
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
sys.path.insert(0, os.path.join(project_root, 'src'))

from imu_calculation import calculate_position_from_imu
from src.mcap_utils import readmcap
from src.handpose3d import handpose3d

config_path = "./configs/ego_config.json"
mcap_path = "/home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap"
mcap_path = '/Users/yangxu/Documents/Code/ff9e3e1189504041b9ce21256925377f.mcap'
output_mp4 = "./processed_data/"
output_mcap_path = "./mcap_output/hand_keypoints.mcap"

mode = "process"  # "process" or "visualize"

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
        handpose3d(input_streams, output_mcap_path, cam_3d_ids=[1, 4], imu_pts=imu_pts, timestamps=timestamps, visualize=False)
