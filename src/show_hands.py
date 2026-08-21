import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import mediapipe as mp
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2
from utils import DLT

NUM_HAND_KEYPOINTS = 21
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = tuple(mp_hands.HAND_CONNECTIONS)
LEFT_HAND_COLOR = (0, 0, 255)
RIGHT_HAND_COLOR = (255, 0, 0)
frame_shape = [1300, 1600]


def read_keypoints(filename, point_dim=3):
    with open(filename, 'r') as fin:
        kpts = []
        while True:
            line = fin.readline()
            if line == '':
                break

            line = line.split()
            if not line:
                continue

            line = [float(s) for s in line]
            values_per_hand = NUM_HAND_KEYPOINTS * point_dim
            if len(line) % values_per_hand != 0:
                raise ValueError(
                    f'Could not reshape {filename}: each frame must contain a multiple of '
                    f'{values_per_hand} values for {point_dim}D hand keypoints.'
                )

            num_hands = len(line) // values_per_hand
            line = np.reshape(line, (num_hands, NUM_HAND_KEYPOINTS, point_dim))
            kpts.append(line)

    return np.array(kpts)


def hand_points_to_mp_landmarks(hand_keypoints, frame_shape):
    hand_keypoints = np.asarray(hand_keypoints, dtype=float)
    if hand_keypoints.shape != (NUM_HAND_KEYPOINTS, 2):
        raise ValueError(
            f'Expected hand keypoints with shape ({NUM_HAND_KEYPOINTS}, 2), '
            f'got {hand_keypoints.shape}.'
        )

    frame_height, frame_width = frame_shape[:2]
    landmarks = []

    for point in hand_keypoints:
        if np.any(point == -1):
            landmarks.append(
                landmark_pb2.NormalizedLandmark(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    visibility=0.0,
                    presence=0.0,
                )
            )
            continue

        landmarks.append(
            landmark_pb2.NormalizedLandmark(
                x=float(point[0]) / frame_width,
                y=float(point[1]) / frame_height,
                z=0.0,
                visibility=1.0,
                presence=1.0,
            )
        )

    return landmark_pb2.NormalizedLandmarkList(landmark=landmarks)


def _draw_hand_2d(frame, hand_keypoints, color):
    hand_keypoints = np.asarray(hand_keypoints)
    if hand_keypoints.shape != (NUM_HAND_KEYPOINTS, 2):
        raise ValueError(
            f'Expected hand keypoints with shape ({NUM_HAND_KEYPOINTS}, 2), '
            f'got {hand_keypoints.shape}.'
        )

    if np.all(hand_keypoints[:, 0] == -1):
        return

    hand_landmarks = hand_points_to_mp_landmarks(hand_keypoints, frame.shape)
    drawing_spec = mp_drawing.DrawingSpec(color=color)
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        drawing_spec,
        drawing_spec,
    )


def visualize_2d(video_path0, video_path1, handpoints0, handpoints1):
    cap0 = cv.VideoCapture(str(video_path0))
    cap1 = cv.VideoCapture(str(video_path1))
    caps = [cap0, cap1]

    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    frame_index = 0
    for frame_handpoints0, frame_handpoints1 in zip(handpoints0, handpoints1):
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        frame0_handpoints = np.asarray(frame_handpoints0, dtype=float)
        frame1_handpoints = np.asarray(frame_handpoints1, dtype=float)

        for hand_idx, hand_keypoints in enumerate(frame0_handpoints):
            print(f"frame shape {frame0.shape}")
            if np.all(hand_keypoints[:, 0] == -1):
                continue

            hand_landmarks = hand_points_to_mp_landmarks(hand_keypoints, frame0.shape)

            if frame_index == 10:
                print(f"hand keypoints {hand_keypoints}")
                print(f"hand landmarks {hand_landmarks}")
            color = LEFT_HAND_COLOR if hand_idx == 0 else RIGHT_HAND_COLOR
            mp_drawing.draw_landmarks(
                frame0,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color),
            )

        for hand_idx, hand_keypoints in enumerate(frame1_handpoints):
            if np.all(hand_keypoints[:, 0] == -1):
                continue

            hand_landmarks = hand_points_to_mp_landmarks(hand_keypoints, frame1.shape)
            color = LEFT_HAND_COLOR if hand_idx == 0 else RIGHT_HAND_COLOR
            mp_drawing.draw_landmarks(
                frame1,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color),
            )

        cv.imshow('Camera 0', frame0)
        cv.imshow('Camera 1', frame1)

        k = cv.waitKey(1) # Wait for 29 ms between frames
        if k & 0xFF == 27:  # ESC key to exit
            break

        frame_index += 1

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


def visualize_3d(p3ds):
    output_dir = Path('figs')
    output_dir.mkdir(exist_ok=True)

    """this contains 3d points of each frame"""
    p3ds = np.array(p3ds)

    """Now visualize in 3D"""
    thumb_f = [[0,1],[1,2],[2,3],[3,4]]
    index_f = [[0,5],[5,6],[6,7],[7,8]]
    middle_f = [[0,9],[9,10],[10,11],[11, 12]]
    ring_f = [[0,13],[13,14],[14,15],[15,16]]
    pinkie_f = [[0,17],[17,18],[18,19],[19,20]]
    fingers = [pinkie_f, ring_f, middle_f, index_f, thumb_f]
    fingers_colors = ['blue', 'blue', 'blue', 'blue', 'blue']

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60, azim=0)

    for i, frame_hands in enumerate(p3ds):
        if i%2 == 0: continue #skip every 2nd frame
        for hand_kpts in frame_hands:
            if np.all(hand_kpts[:, 0] == -1):
                continue

            for finger, finger_color in zip(fingers, fingers_colors):
                for _c in finger:
                    if np.any(hand_kpts[_c[0]] == -1) or np.any(hand_kpts[_c[1]] == -1):
                        continue

                    ax.plot(
                        xs=[hand_kpts[_c[0], 0], hand_kpts[_c[1], 0]],
                        ys=[hand_kpts[_c[0], 1], hand_kpts[_c[1], 1]],
                        zs=[hand_kpts[_c[0], 2], hand_kpts[_c[1], 2]],
                        linewidth=4,
                        c=finger_color,
                    )
                    ax.set_box_aspect([1, 1, 1])
                    # ax.view_init(elev=30, azim=-75) 

        #draw axes
        ax.plot(xs = [0,1], ys = [0,0], zs = [0,0], linewidth = 2, color = 'red')
        ax.plot(xs = [0,0], ys = [0,1], zs = [0,0], linewidth = 2, color = 'blue')
        ax.plot(xs = [0,0], ys = [0,0], zs = [0,1], linewidth = 2, color = 'black')

        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-0.50, 0.5)
        ax.set_xlabel('x')
        ax.set_ylim3d(-0.50, 0.5)
        ax.set_ylabel('y')
        ax.set_zlim3d(-0.50, 0.5)
        ax.set_zlabel('z')
        # plt.savefig(output_dir / f'fig_{i}.png')
        plt.pause(0.01)
        ax.cla()


if __name__ == '__main__':
    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
