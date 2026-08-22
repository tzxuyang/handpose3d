import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import mediapipe as mp
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2
from utils import configure_hand_3d_axes, draw_hand_3d, transform_hand_ninety_degree

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
            if np.all(hand_keypoints[:, 0] == -1):
                continue

            hand_landmarks = hand_points_to_mp_landmarks(hand_keypoints, frame0.shape)

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

    p3ds = np.array(p3ds)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)

    for i, frame_hands in enumerate(p3ds):
        # frame_hands = transform_hand_ninety_degree(frame_hands)
        for hand_kpts in frame_hands:
            draw_hand_3d(ax, hand_kpts, color='blue', marker='o')

        configure_hand_3d_axes(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'Generated handpoints\nframe={i}')
        plt.pause(0.01)
        ax.cla()


if __name__ == '__main__':
    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
