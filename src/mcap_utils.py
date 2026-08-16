#!/usr/bin/python3.11
"""
mcap → MP4 视频提取工具

从 EgoScale mcap 文件中提取指定相机的 H.264 视频流并转换为 MP4。

用法:
    # 提取单个相机
    python3 extract_video.py <mcap文件> <相机topic> -o 输出.mp4

    # 提取所有相机（默认 camera0~5）
    python3 extract_video.py <mcap文件> --all -o 输出目录/

    # 指定 FPS
    python3 extract_video.py <mcap文件> <topic> -o 输出.mp4 --fps 30

示例:
    python3 extract_video.py data.mcap /robot0/sensor/camera0/compressed -o cam0.mp4
    python3 extract_video.py data.mcap --all -o ./videos/
"""
import json
import os
import sys
import argparse
import subprocess
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
from mcap_protobuf.decoder import DecoderFactory
from google.protobuf.json_format import MessageToDict
from google.protobuf import descriptor_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
from mcap.writer import Writer
from genson import SchemaBuilder
import numpy as np
from utils import write_extrinsic_parameters, write_instrinsic_parameters, quat_to_rot_matrix, msg_time_sync

DISTORTION = [-0.0, 0.0, 0.0, 0.0, 0.0] # Assuming no distortion for simplicity, replace with actual values if needed

BONE_NAMES = [
    "Hand",
    "Thumb0", "Thumb1", "Thumb2", "Thumb3",
    "Index0", "Index1", "Index2", "Index3",
    "Middle0", "Middle1", "Middle2", "Middle3",
    "Ring0", "Ring1", "Ring2", "Ring3",
    "Pinkie0", "Pinkie1", "Pinkie2", "Pinkie3"
]

def generate_jsonschema_2d():
    """Generates a dynamic JSON Schema matching your hand data layout."""
    point_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["x", "y"]
    }
    
    hand_properties = {
        str(i): {
            "type": "object",
            "properties": {
                "point": point_schema
            },
            "required": ["point"]
        } for i in range(21)
    }
    
    keypoints_schema = {
        "type": "object",
        "properties": hand_properties,
        "required": [str(i) for i in range(21)]
    }

    hands_properties = {
        str(i): {
            "type": "object",
            "properties": {
                "keypoints": keypoints_schema
            },
            "required": ["keypoints"]
        } for i in range(2)  # 0: Left, 1: Right
    }
    return {
        "type": "object",
        "properties": {
            "header": {
                "type": "object",
                "properties": {"timestamp": {"type": "number"}, "frame_index": {"type": "integer"}, "topic_name": {"type": "string"}},
                "required": ["timestamp", "frame_index", "topic_name"]
            },
            "hands": {"type": "object", "properties": hands_properties, "required": [str(i) for i in range(2)]}
        },
        "required": ["header", "hands"]
    }


def _to_builtin_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value

def generate_jsonschema_3d():
    """Generates a dynamic JSON Schema matching your hand data layout."""
    pos_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number"}, 
            "y": {"type": "number"}, 
            "z": {"type": "number"}
        },
        "required": ["x", "y", "z"]
    }
    bone_properties = {
        str(i): {
            "type": "object",
            "properties": {
                "bone_name": {"type": "string"},
                "to_global": {
                    "type": "object", 
                    "properties": {"position": pos_schema}, 
                    "required": ["position"]
                }
            },
            "required": ["bone_name", "to_global"]
        } for i in range(21)
    }
    return {
        "type": "object",
        "properties": {
            "header": {
                "type": "object",
                "properties": {"timestamp": {"type": "number"}, "frame_index": {"type": "integer"}, "topic_name": {"type": "string"}},
                "required": ["timestamp", "frame_index", "topic_name"]
            },
            "bone_data": {"type": "object", "properties": bone_properties, "required": [str(i) for i in range(21)]}
        },
        "required": ["header", "bone_data"]
    }

def build_hand_message_2d(timestamp, idx, handpoints, topic_name):
    """Formats raw array parameters into your exact dictionary schema."""
    hands = {}

    handpoints_list = []
    for j in range(2):  # 0: Left, 1: Right
        temp_list = []
        for i in range(21):
            vx = int(_to_builtin_scalar(handpoints[j][i][0]))
            vy = int(_to_builtin_scalar(handpoints[j][i][1]))
            temp_list.append([vx, vy])
        handpoints_list.append(temp_list)

    for j in range(2):  # 0: Left, 1: Right
        hands[str(j)] = {
            "keypoints": {
                str(i): {
                    "point": {
                        "x": _to_builtin_scalar(handpoints_list[j][i][0]),
                        "y": _to_builtin_scalar(handpoints_list[j][i][1])
                    }
                } for i in range(21)
            }
        }

    return {
        "header": {
            "timestamp": timestamp,
            "frame_index": idx,
            "topic_name": topic_name
        },
        "hands": hands
    }

def build_hand_message_3d(timestamp, idx, left_right, handpoints):
    """Formats raw array parameters into your exact dictionary schema."""
    prefix = left_right.capitalize()
    bone_data = {}
    
    for i, name in enumerate(BONE_NAMES):
        bone_data[str(i)] = {
            "bone_name": f"{prefix}_{name}",
            "to_global": {
                "position": {
                    "x": handpoints[i][0],
                    "y": handpoints[i][1],
                    "z": handpoints[i][2]
                }
            }
        }
        
    return {
        "header": {
            "timestamp": timestamp,
            "frame_index": idx,
            "topic_name": f"/robot0/handtracking/{left_right}"
        },
        "bone_data": bone_data
    }

def construct_2d_hand_keypoints_msg(kpts_2d, timestamps, topic_name):
    """Constructs a list of dictionaries for left and right hand keypoints."""
    hands_msgs = []
    
    for idx, (kpts, timestamp) in enumerate(zip(kpts_2d, timestamps)):
        hands_msg = build_hand_message_2d(timestamp, idx, kpts, topic_name)        
        hands_msgs.append(hands_msg)
    
    return hands_msgs

def construct_3d_hand_keypoints_msg(kpts_3d, timestamps):
    """Constructs a list of dictionaries for left and right hand keypoints."""
    left_hand_msgs = []
    right_hand_msgs = []
    
    for idx, (kpts, timestamp) in enumerate(zip(kpts_3d, timestamps)):
        left_hand_msg = build_hand_message_3d(timestamp, idx, "left", kpts[0])
        right_hand_msg = build_hand_message_3d(timestamp, idx, "right", kpts[1])
        
        left_hand_msgs.append(left_hand_msg)
        right_hand_msgs.append(right_hand_msg)
    
    return left_hand_msgs, right_hand_msgs

def write_2d_hand_keypoints_mcap(kpts_2d, timestamps, topic_name, output_mcap_path):
    """Writes the 2D hand keypoints to an MCAP file (creates/overwrites file)."""
    
    with open(output_mcap_path, "wb") as f:
        writer = Writer(f)
        writer.start()

        # 1. Register Schema rules
        schema_id = writer.register_schema(
            name="robot_hand_tracking",
            encoding="jsonschema",
            data=json.dumps(generate_jsonschema_2d()).encode("utf-8")
        )

        # 2. Register separate channels for Left and Right topics
        hands_channel = writer.register_channel(
            topic=topic_name,
            message_encoding="json",
            schema_id=schema_id
        )

        # 3. Write messages for each timestamp
        for idx, (kpts, timestamp) in enumerate(zip(kpts_2d, timestamps)):
            hands_msg = build_hand_message_2d(timestamp, idx, kpts, topic_name)

            time_ns = int(timestamp)

            # 5. Write messages to the file structure
            writer.add_message(
                channel_id=hands_channel,
                log_time=time_ns,
                publish_time=time_ns,
                sequence=0,
                data=json.dumps(hands_msg).encode("utf-8")
            )

        writer.finish()

def write_3d_hand_keypoints_mcap(kpts_3d, timestamps, output_mcap_path):
    """Writes the 3D hand keypoints to an MCAP file."""
    
    with open(output_mcap_path, "wb") as f:
        writer = Writer(f)
        writer.start()

        # 1. Register Schema rules
        schema_id = writer.register_schema(
            name="robot_hand_tracking",
            encoding="jsonschema",
            data=json.dumps(generate_jsonschema_3d()).encode("utf-8")
        )

        # 2. Register separate channels for Left and Right topics
        left_channel = writer.register_channel(
            topic="/robot0/handtracking/left",
            message_encoding="json",
            schema_id=schema_id
        )
        right_channel = writer.register_channel(
            topic="/robot0/handtracking/right",
            message_encoding="json",
            schema_id=schema_id
        )

        # 3. Write messages for each timestamp
        for idx, (kpts, timestamp) in enumerate(zip(kpts_3d, timestamps)):
            left_hand_msg = build_hand_message_3d(timestamp, idx, "left", kpts[0])
            right_hand_msg = build_hand_message_3d(timestamp, idx, "right", kpts[1])

            time_ns = int(timestamp)

            # 5. Write messages to the file structure
            writer.add_message(
                channel_id=left_channel,
                log_time=time_ns,
                publish_time=time_ns,
                sequence=0,
                data=json.dumps(left_hand_msg).encode("utf-8")
            )
            writer.add_message(
                channel_id=right_channel,
                log_time=time_ns,
                publish_time=time_ns,
                sequence=0,
                data=json.dumps(right_hand_msg).encode("utf-8")
            )
            
        writer.finish()

def safe_merge_mcaps(input_paths, output_path):
    """Safely merge multiple MCAP files into a single MCAP by rewriting messages.

    This avoids raw byte concatenation which corrupts MCAP files. Each input file is read
    using make_reader and all messages are replayed into a new Writer.
    """
    # normalize paths
    input_paths = [str(p) for p in input_paths]

    with open(output_path, "wb") as f_out:
        writer = Writer(f_out)
        writer.start()

        schema_map = {}
        channel_map = {}

        for p in input_paths:
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f_in:
                reader = make_reader(f_in)
                for schema, channel, message in reader.iter_messages():
                    # register schema if unknown
                    schema_key = getattr(schema, "name", None) or getattr(schema, "encoding", "schema")
                    if schema_key not in schema_map:
                        # Some schema objects expose .data as bytes; fall back to empty bytes if absent
                        schema_data = getattr(schema, "data", None)
                        if schema_data is None and hasattr(schema, "json_schema"):
                            schema_data = schema.json_schema
                        if schema_data is None:
                            schema_data = b""
                        schema_id = writer.register_schema(
                            name=getattr(schema, "name", ""),
                            encoding=getattr(schema, "encoding", ""),
                            data=schema_data
                        )
                        schema_map[schema_key] = schema_id
                    else:
                        schema_id = schema_map[schema_key]

                    # register channel if unknown
                    chan_key = channel.topic
                    if chan_key not in channel_map:
                        ch_id = writer.register_channel(
                            topic=channel.topic,
                            message_encoding=getattr(channel, "message_encoding", ""),
                            schema_id=schema_id
                        )
                        channel_map[chan_key] = ch_id
                    else:
                        ch_id = channel_map[chan_key]

                    # re-add message
                    writer.add_message(
                        channel_id=ch_id,
                        log_time=message.log_time,
                        publish_time=getattr(message, "publish_time", message.log_time),
                        sequence=getattr(message, "sequence", 0),
                        data=message.data
                    )

        writer.finish()


def readmcap(mcap_path, config_path, output_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        video_topics = config.get("camera_topics")
        cam_info = config.get("camera_info")
        imu_topic = config.get("imu_topic")

    # Extract video from MCAP file
    extract_video(mcap_path, video_topics, output_path, fps=30, keep_h264=False)

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
    write_instrinsic_parameters("./camera_parameters/c0.dat", K0, DISTORTION)
    write_instrinsic_parameters("./camera_parameters/c1.dat", K1, DISTORTION)
    write_extrinsic_parameters("./camera_parameters/rot_trans_c0.dat", R0, T0)
    write_extrinsic_parameters("./camera_parameters/rot_trans_c1.dat", R1, T1)

    # Read IMU data
    msg_cam = read_mcap_protobuf(mcap_path, video_topics[0])
    msg_imu = read_mcap_protobuf(mcap_path, imu_topic)
    msg_imu_synced = msg_time_sync(msg_cam, msg_imu)

    timestamps = [msg['header']['timestamp'] for msg in msg_cam]
    
    return msg_imu_synced, timestamps, height, width

def read_mcap_protobuf_once(mcap_path, topics=None):
    with open(mcap_path, "rb") as f:
        # Pass DecoderFactory to automatically unpack Protobuf schemas
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        
        # Iterate over all messages across all topics
        for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=topics):            
            # proto_msg is a fully hydrated Python Protobuf object
            return proto_msg  # You can process or print the proto_msg as needed

def read_mcap_protobuf(mcap_path, topics=None):
    messages_dict = []
    with open(mcap_path, "rb") as f:
        # Pass DecoderFactory to automatically unpack Protobuf schemas
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        
        # Iterate over all messages across all topics
        for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=topics):            
            # proto_msg is a fully hydrated Python Protobuf object
            if proto_msg is None:
                continue
                
            # Convert the Protobuf object to a standard Python dictionary
            msg_dict = MessageToDict(
                proto_msg,
                preserving_proto_field_name=True  # Keeps original .proto snake_case names
            )
                         
            messages_dict.append(msg_dict)
            
    return messages_dict
        
def read_mcap_topics(mcap_path, topics=None):
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=topics):
            print(f"Topic: {channel.topic} | Schema: {schema.name} | Log Time: {message.log_time} | Data Size: {len(message.data)}")
            print(f"Timestamp: {message.log_time} | Data size: {len(message.data)}")
            print(f"Data (first 10 bytes): {message.data[:10]}")

def read_varint(data, offset):
    result = shift = 0
    while offset < len(data):
        byte = data[offset]
        result |= (byte & 0x7F) << shift
        offset += 1
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result, offset

def extract_h264_data(msg_data):
    """从 foxglove.CompressedImage protobuf 消息中提取 H.264 数据"""
    offset = 0
    while offset < len(msg_data):
        tag, offset = read_varint(msg_data, offset)
        fn = tag >> 3
        wt = tag & 0x7
        if fn == 2 and wt == 2:  # data 字段
            length, offset = read_varint(msg_data, offset)
            return msg_data[offset:offset + length]
        elif wt == 2:
            length, offset = read_varint(msg_data, offset)
            offset += length
        elif wt == 0:
            _, offset = read_varint(msg_data, offset)
        elif wt in (1, 5):
            offset += 8 if wt == 1 else 4
        else:
            break
    return None


def extract_to_h264(mcap_path, topic, output_h264):
    """从 mcap 提取指定 topic 的 H.264 裸流"""
    frame_count = 0
    total_bytes = 0

    with open(mcap_path, 'rb') as f_in:
        reader = make_reader(f_in)
        with open(output_h264, 'wb') as f_out:
            for schema, channel, msg in reader.iter_messages():
                if channel.topic == topic:
                    h264 = extract_h264_data(msg.data)
                    if h264:
                        f_out.write(h264)
                        frame_count += 1
                        total_bytes += len(h264)
    return frame_count, total_bytes


def h264_to_mp4(h264_path, mp4_path, fps=30, reencode=False):
    """用 ffmpeg 将 H.264 裸流转为 MP4"""
    if reencode:
        cmd = [
            'ffmpeg', '-y', '-f', 'h264', '-r', str(fps),
            '-i', h264_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            mp4_path
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-f', 'h264', '-r', str(fps),
            '-i', h264_path,
            '-c', 'copy',
            '-movflags', '+faststart',
            mp4_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # 直接 copy 失败，尝试重新编码
        if not reencode:
            print(f"    ⚠️  remux 失败, 尝试重新编码...")
            return h264_to_mp4(h264_path, mp4_path, fps, reencode=True)
        else:
            print(f"    ❌ ffmpeg 错误: {result.stderr[:300]}")
            return False
    return True


def extract_single(mcap_path, topic, output_mp4, fps=30, keep_h264=False):
    """提取单个相机为 MP4"""
    h264_path = output_mp4.replace('.mp4', '.h264')

    print(f"📷 {topic}")
    frame_count, total_bytes = extract_to_h264(mcap_path, topic, h264_path)

    if frame_count == 0:
        print(f"    ❌ 未找到数据")
        return False

    print(f"    提取 {frame_count} 帧 ({total_bytes / 1024 / 1024:.1f} MB)")

    if not h264_to_mp4(h264_path, output_mp4, fps):
        return False

    if not keep_h264:
        os.remove(h264_path)

    mp4_size = os.path.getsize(output_mp4) / (1024 * 1024)
    print(f"    ✅ {output_mp4} ({mp4_size:.1f} MB)")
    return True


def list_topics(mcap_path):
    """列出 mcap 中所有相机 topic"""
    topics = []
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary and summary.channels:
            for cid, chan in summary.channels.items():
                if 'compressed' in chan.topic and 'camera' in chan.topic:
                    topics.append(chan.topic)
    return sorted(topics)

def extract_video(mcap_path, topics, output_mp4, fps=30, keep_h264=False):
    if not os.path.exists(mcap_path):
        print(f"❌ 文件不存在: {mcap_path}")
        sys.exit(1)


    # 提取所有相机
    if topics == 'all':
        topics = list_topics(mcap_path)
        if not topics:
            print("❌ 未找到任何相机 topic")
            sys.exit(1)

        out_dir = output_mp4 or os.path.splitext(mcap_path)[0] + '_videos'
        os.makedirs(out_dir, exist_ok=True)

        print(f"🎬 提取 {len(topics)} 路相机 → {out_dir}/\n")
        for topic in topics:
            cam_name = topic.split('/')[-2]  # camera0, camera1, ...
            out_mp4 = os.path.join(out_dir, f'{cam_name}.mp4')
            extract_single(mcap_path, topic, out_mp4, fps, keep_h264)
            print()
        print(f"✅ 全部完成! 视频目录: {out_dir}/")
        # sys.exit(0)

    # 提取单个相机
    elif topics:
        out_dir = output_mp4 or os.path.splitext(mcap_path)[0] + '_videos'
        os.makedirs(out_dir, exist_ok=True)

        topics = [topics] if isinstance(topics, str) else topics
        for topic in topics:
            cam_name = topic.split('/')[-2]  # camera0, camera1, ...
            out_mp4 = os.path.join(out_dir, f'{cam_name}.mp4')
            extract_single(mcap_path, topic, out_mp4, fps, keep_h264)
        print(f"✅ 全部完成! 视频目录: {out_dir}/")
        # sys.exit(0)

    else:
        print("❌ 请指定相机 topic 或使用 'all'")
        print("   提示: 用 list_topics() 查看可用 topic")
        # sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='从 EgoScale mcap 文件中提取 H.264 视频为 MP4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('mcap', help='mcap 文件路径')
    parser.add_argument('topic', nargs='?', default=None,
                        help='相机 topic (如 /robot0/sensor/camera0/compressed)')
    parser.add_argument('-o', '--output', default=None,
                        help='输出 MP4 文件路径（或 --all 模式下的输出目录）')
    parser.add_argument('--all', action='store_true',
                        help='提取所有相机')
    parser.add_argument('--fps', type=int, default=30,
                        help='帧率 (默认: 30)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有相机 topic 后退出')
    parser.add_argument('--keep-h264', action='store_true',
                        help='保留中间 H.264 裸流文件')

    args = parser.parse_args()

    if not os.path.exists(args.mcap):
        print(f"❌ 文件不存在: {args.mcap}")
        sys.exit(1)

    # 列出 topic
    if args.list:
        topics = list_topics(args.mcap)
        print(f"📋 {args.mcap} 中的相机 topic:")
        for t in topics:
            print(f"    {t}")
        sys.exit(0)

    # 提取所有相机
    if args.all:
        topics = list_topics(args.mcap)
        if not topics:
            print("❌ 未找到任何相机 topic")
            sys.exit(1)

        out_dir = args.output or os.path.splitext(args.mcap)[0] + '_videos'
        os.makedirs(out_dir, exist_ok=True)

        print(f"🎬 提取 {len(topics)} 路相机 → {out_dir}/\n")
        for topic in topics:
            cam_name = topic.split('/')[-2]  # camera0, camera1, ...
            out_mp4 = os.path.join(out_dir, f'{cam_name}.mp4')
            extract_single(args.mcap, topic, out_mp4, args.fps, args.keep_h264)
            print()
        print(f"✅ 全部完成! 视频目录: {out_dir}/")
        sys.exit(0)

    # 提取单个相机
    if not args.topic:
        print("❌ 请指定相机 topic 或使用 --all")
        print("   提示: 用 --list 查看可用 topic")
        sys.exit(1)

    out_mp4 = args.output or os.path.splitext(os.path.basename(args.mcap))[0] + '.mp4'
    extract_single(args.mcap, args.topic, out_mp4, args.fps, args.keep_h264)


if __name__ == '__main__':
    main()
    # uv run src/extract_video.py /home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap --all -o ./processed_data/