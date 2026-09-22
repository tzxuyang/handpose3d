from dataclasses import dataclass
import os
import sys
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
sys.path.insert(0, os.path.join(project_root, 'src'))

from imu_calculation import calculate_position_from_imu
from mcap_utils import read_hand_json_2d, read_hand_json_3d, read_hand_message_2d, read_mcap_json, read_mcap_protobuf, readmcap, read_hand_message_3d
from handpose3d import handpose3d
from show_hands import visualize_2d, visualize_3d
from vio.config import VioConfig
from vio.pipeline import start_vio_task
import tyro
import numpy as np

@dataclass
class Config:
    
    # mcap_path: str = "/home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap"
    mcap_path: str = "/home/ubuntu/handpose3D_test/groundTruth_data/ff9e3e1189504041b9ce21256925377f.mcap"
    mode: str = "visualize"  # "process" or "visualize"
    visualize_mcap_path: str = "./mcap_output/hand_keypoints.mcap"
    world_mcap_path: str = "./mcap_output/hand_keypoints_world.mcap"
    skip_vio: bool = False
    vio_binary: str = ""
    vio_output_dir: str = ""

config_path = "./configs/ego_config.json"
# mcap_path = "/home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap"
# mcap_path = "/home/yang/Downloads/0fa832d8c9814bd98ca33461f24a8bbf.mcap"
# mcap_path = '/Users/yangxu/Documents/Code/ff9e3e1189504041b9ce21256925377f.mcap'
output_mp4 = "./processed_data/"
output_mcap_path = "./mcap_output/hand_keypoints.mcap"

mode = "visualize"  # "process" or "visualize"


def run_process(config: Config) -> dict:
    """``--mode process``: extract, detect, triangulate and fuse a VIO trajectory."""
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # start the VIO stage first: it only needs cameras 1/4 + IMU and runs
    # concurrently with the MediaPipe pass, which takes much longer.
    vio_config = VioConfig.from_ego_config(
        config_path,
        mcap_path=config.mcap_path,
        enabled=not config.skip_vio,
        basalt_bin=config.vio_binary or None,
        output_dir=config.vio_output_dir or None,
        hand_mcap=output_mcap_path,
    )
    if config.skip_vio:
        print("[vio] disabled by --skip-vio")
    vio_task = start_vio_task(vio_config)

    # read the input mcap file and extract camera videos
    msg_imu_synced, timestamps, height, width = readmcap(config.mcap_path, config_path, output_mp4)

    # detect hand keypoints in 2D and 3D using mediapipe for each camera frame
    imu_pts = 1
    input_streams = [f'processed_data/camera{i}.mp4' for i in range(6)]
    summary = handpose3d(
        input_streams,
        output_mcap_path,
        cam_3d_ids=[1, 4],
        imu_pts=imu_pts,
        timestamps=timestamps,
        visualize=True,
        pose_task=vio_task,
        world_mcap_path=config.world_mcap_path,
        qc_path=str(vio_config.resolved_qc_path),
    )
    print(json.dumps(summary, indent=2))
    return summary


def run_visualize(config: Config) -> None:
    """``--mode visualize``: replay the 2D/3D keypoints stored in an mcap."""
    with open(config_path, "r") as f:
        config_data = json.load(f)
        handpoints_3d_topic = config_data.get("handpoints_3d_topic")
        handpoints_2d_topic = config_data.get("handpoints_2d_topic")

    # read the input mcap file of camera videos and visualize the 2D hand keypoints
    video_stream_0 = 'processed_data/camera1.mp4'
    video_stream_1 = 'processed_data/camera4.mp4'
    msg_2d_0 = read_mcap_json(config.visualize_mcap_path, handpoints_2d_topic[1])
    msg_2d_1 = read_mcap_json(config.visualize_mcap_path, handpoints_2d_topic[4])

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
    msg_left = read_mcap_json(config.visualize_mcap_path, handpoints_3d_topic[0])
    msg_right = read_mcap_json(config.visualize_mcap_path, handpoints_3d_topic[1])

    p3ds = []
    for frame_left, frame_right in zip(msg_left, msg_right):
        left_kpts = read_hand_json_3d(frame_left)
        right_kpts = read_hand_json_3d(frame_right)
        p3ds.append([left_kpts, right_kpts])

    visualize_3d(p3ds)


if __name__ == "__main__":
    cli_config = tyro.cli(Config)

    if cli_config.mode == "process":
        run_process(cli_config)
    elif cli_config.mode == "visualize":
        run_visualize(cli_config)
