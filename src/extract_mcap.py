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

import os
import sys
import argparse
import subprocess
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages
from mcap_protobuf.decoder import DecoderFactory

def read_mcap_protobuf(mcap_path, topics=None):
    with open(mcap_path, "rb") as f:
        # Pass DecoderFactory to automatically unpack Protobuf schemas
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        
        # Iterate over all messages across all topics
        for schema, channel, message, proto_msg in reader.iter_decoded_messages(topics=topics):            
            # proto_msg is a fully hydrated Python Protobuf object
            return proto_msg  # You can process or print the proto_msg as needed

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