from dataclasses import dataclass
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.lines import Line2D
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))

from utils import configure_hand_3d_axes, draw_hand_3d, transform_hand_ninety_degree

HAND_TOPICS = ("/robot0/handtracking/left", "/robot0/handtracking/right")


@dataclass
class Config:
    generated_mcap_path: str = "./mcap_output/hand_keypoints.mcap"
    ground_truth_mcap_path: str = "/home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap"
    pause: float = 0.03
    stride: int = 1
    max_frames: int | None = None


def _read_generated_topic(mcap_path, topic_name):
    frames = []
    with open(mcap_path, "rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages(topics=[topic_name]):
            payload = json.loads(message.data.decode("utf-8"))
            points = []
            for point_idx in range(21):
                position = payload["bone_data"][str(point_idx)]["to_global"]["position"]
                points.append([position["x"], position["y"], position["z"]])
            frames.append((int(payload["header"]["timestamp"]), np.asarray(points, dtype=np.float32)))
    return frames


def _read_ground_truth_topic(mcap_path, topic_name):
    frames = []
    with open(mcap_path, "rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=[topic_name]):
            points = []
            for bone in proto_msg.bone_data:
                position = bone.to_global.position
                points.append([position.x, position.y, position.z])
            frames.append((int(message.log_time), np.asarray(points, dtype=np.float32)))
    return frames


def _read_hand_frames(mcap_path, reader_fn):
    left_frames = reader_fn(mcap_path, HAND_TOPICS[0])
    right_frames = reader_fn(mcap_path, HAND_TOPICS[1])
    if len(left_frames) != len(right_frames):
        raise ValueError(
            f"Mismatched frame counts between {HAND_TOPICS[0]} ({len(left_frames)}) "
            f"and {HAND_TOPICS[1]} ({len(right_frames)})."
        )

    combined_frames = []
    for (left_ts, left_points), (right_ts, right_points) in zip(left_frames, right_frames):
        if left_ts != right_ts:
            raise ValueError(f"Left/right hand timestamps do not match: {left_ts} vs {right_ts}.")
        combined_frames.append((left_ts, np.stack([left_points, right_points], axis=0)))

    return combined_frames


def _find_nearest_frame(timestamp, frames, frame_timestamps):
    insert_index = int(np.searchsorted(frame_timestamps, timestamp))
    candidate_indices = []
    if insert_index < len(frames):
        candidate_indices.append(insert_index)
    if insert_index > 0:
        candidate_indices.append(insert_index - 1)

    best_index = min(candidate_indices, key=lambda idx: abs(int(frame_timestamps[idx]) - timestamp))
    return frames[best_index]

def build_comparison_frames(generated_mcap_path, ground_truth_mcap_path):
    generated_frames = _read_hand_frames(generated_mcap_path, _read_generated_topic)
    ground_truth_frames = _read_hand_frames(ground_truth_mcap_path, _read_ground_truth_topic)
    ground_truth_timestamps = np.asarray([timestamp for timestamp, _ in ground_truth_frames], dtype=np.int64)

    comparison_frames = []
    for generated_timestamp, generated_points in generated_frames:
        matched_timestamp, ground_truth_points = _find_nearest_frame(
            generated_timestamp,
            ground_truth_frames,
            ground_truth_timestamps,
        )
        # generated_points = transform_hand_ninety_degree(generated_points)
        comparison_frames.append(
            {
                "generated_timestamp": generated_timestamp,
                "ground_truth_timestamp": matched_timestamp,
                "timestamp_delta_ns": abs(matched_timestamp - generated_timestamp),
                "generated_points": generated_points,
                "ground_truth_points": ground_truth_points,
            }
        )

    return comparison_frames


def visualize_comparison(comparison_frames, pause, stride, max_frames=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=-60)

    legend_handles = [
        Line2D([0], [0], color="blue", marker="o", linewidth=2, label="Generated"),
        Line2D([0], [0], color="red", marker="x", linewidth=2, label="Ground truth"),
    ]

    plotted_frames = comparison_frames[::max(stride, 1)]
    if max_frames is not None:
        plotted_frames = plotted_frames[:max_frames]

    for frame_index, frame in enumerate(plotted_frames):
        ax.cla()
        ax.legend(handles=legend_handles, loc="upper right")

        generated_points = frame["generated_points"]
        ground_truth_points = frame["ground_truth_points"]

        for hand_points in generated_points:
            draw_hand_3d(ax, hand_points, color="blue", marker="o")
        for hand_points in ground_truth_points:
            draw_hand_3d(ax, hand_points, color="red", marker="x")

        configure_hand_3d_axes(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(
            "Generated (blue) vs ground truth (red)\n"
            f"frame={frame_index} generated_ts={frame['generated_timestamp']} "
            f"gt_ts={frame['ground_truth_timestamp']} "
            f"dt={frame['timestamp_delta_ns'] / 1e6:.3f} ms"
        )
        plt.pause(pause)

    plt.show()


def main():
    config = tyro.cli(Config)
    comparison_frames = build_comparison_frames(
        config.generated_mcap_path,
        config.ground_truth_mcap_path,
    )
    visualize_comparison(
        comparison_frames,
        pause=config.pause,
        stride=config.stride,
        max_frames=config.max_frames,
    )


if __name__ == "__main__":
    main()
