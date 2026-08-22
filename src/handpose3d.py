import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from mcap_utils import construct_2d_hand_keypoints_msg, read_mcap_protobuf, construct_3d_hand_keypoints_msg, write_2d_hand_keypoints_mcap, write_3d_hand_keypoints_mcap, safe_merge_mcaps
from imu_calculation import calculate_position_from_imu
from utils import (
    msg_time_sync,
    write_keypoints_to_disk,
    read_rotation_translation,
    read_camera_parameters,
    triangulate_rays,
    unproject_pixel,
)
from pymcap import PyMCAP
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [1300, 1600]
HAND_LABELS = ('Left', 'Right')
NUM_HAND_KEYPOINTS = 21
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def _empty_frame_keypoints(num_hands, point_dim):
    return [[[-1] * point_dim for _ in range(NUM_HAND_KEYPOINTS)] for _ in range(num_hands)]


def _get_hand_slot(results, detected_index, filled_slots):
    if results.multi_handedness and detected_index < len(results.multi_handedness):
        label = results.multi_handedness[detected_index].classification[0].label
        if label in HAND_LABELS:
            preferred_slot = HAND_LABELS.index(label)
            if preferred_slot not in filled_slots:
                return preferred_slot

    for slot in range(len(HAND_LABELS)):
        if slot not in filled_slots:
            return slot

    return None


def _extract_frame_keypoints(results, frame, point_dim):
    frame_keypoints = _empty_frame_keypoints(len(HAND_LABELS), point_dim)

    if not results.multi_hand_landmarks:
        return frame_keypoints

    filled_slots = set()
    for detected_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        hand_slot = _get_hand_slot(results, detected_index, filled_slots)
        if hand_slot is None:
            continue

        filled_slots.add(hand_slot)
        for p in range(NUM_HAND_KEYPOINTS):
            pxl_x = int(round(frame.shape[1] * hand_landmarks.landmark[p].x))
            pxl_y = int(round(frame.shape[0] * hand_landmarks.landmark[p].y))
            frame_keypoints[hand_slot][p] = [pxl_x, pxl_y]

    return frame_keypoints


def _unproject_hand_keypoints(hand_keypoints, camera_matrix, distortion, distortion_model):
    hand_keypoints = np.asarray(hand_keypoints, dtype=np.float32)
    unprojected_rays = np.full((NUM_HAND_KEYPOINTS, 3), -1.0, dtype=np.float32)

    valid_mask = np.all(hand_keypoints != -1, axis=1)
    if not np.any(valid_mask):
        return unprojected_rays

    for point_idx in np.flatnonzero(valid_mask):
        ray = unproject_pixel(
            hand_keypoints[point_idx],
            camera_matrix,
            distortion,
            distortion_model=distortion_model,
        )
        if ray is None:
            continue

        unprojected_rays[point_idx] = ray.astype(np.float32)

    return unprojected_rays


def _filter_hand_points_3d(hand_points_3d):
    hand_points_3d = np.asarray(hand_points_3d, dtype=np.float32)
    valid_mask = hand_points_3d[:, 0] != -1
    if not np.any(valid_mask):
        return hand_points_3d

    valid_points = hand_points_3d[valid_mask]
    median_depth = np.median(valid_points[:, 2])

    wrist_point = hand_points_3d[0] if hand_points_3d[0, 0] != -1 else None
    filtered_points = hand_points_3d.copy()
    for point_idx, point in enumerate(filtered_points):
        if point[0] == -1:
            continue

        if abs(point[2] - median_depth) > 0.12:
            filtered_points[point_idx] = [-1, -1, -1]
            continue

        if wrist_point is not None and np.linalg.norm(point - wrist_point) > 0.22:
            filtered_points[point_idx] = [-1, -1, -1]

    return filtered_points


def run_mp(input_streams, P0, P1, cam_ids = [1,4], visualize=False):
    #read camera parameters
    cmtx0, dist0, distortion_model0 = read_camera_parameters(0)
    cmtx1, dist1, distortion_model1 = read_camera_parameters(1)
    rmat0, tvec0 = read_rotation_translation(0)
    rmat1, tvec1 = read_rotation_translation(1)
    rmat0 = np.asarray(rmat0, dtype=np.float32).reshape(3, 3)
    rmat1 = np.asarray(rmat1, dtype=np.float32).reshape(3, 3)
    tvec0 = np.asarray(tvec0, dtype=np.float32).reshape(3)
    tvec1 = np.asarray(tvec1, dtype=np.float32).reshape(3)

    #input video stream
    caps = [cv.VideoCapture(input_stream) for input_stream in input_streams]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])
    #create hand keypoints detector object.
    hands = [mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=2, min_tracking_confidence=0.5) for _ in input_streams]

    #containers for detected keypoints for each camera
    kpts_cam = []
    for i in range(len(hands)):
        kpts_cam.append([])
    kpts_3d = []

    frame_idx = 0
    while True:

        #read frames from stream
        ret =[]
        frame = []
        results = []

        for i, cap in enumerate(caps):
            ret_read, frame_read = cap.read()
            ret.append(ret_read)
            frame.append(frame_read)

        if not ret[cam_ids[0]] or not ret[cam_ids[1]]: break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        for i in range(len(input_streams)):
            if frame[i] is not None and frame[i].shape[1] != 720:
                # frame_temp = frame[i][:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
                # the BGR image to RGB.
                frame[i] = cv.cvtColor(frame[i], cv.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                frame[i].flags.writeable = False
                results.append(hands[i].process(frame[i]))
            else:
                results.append(results[-1] if results else None)

        #prepare list of hand keypoints of this frame
        #frame0 kpts
        for i in range(len(input_streams)):
            if frame[i] is not None:
                frame_keypoints = _extract_frame_keypoints(results[i], frame[i], point_dim=2)
                kpts_cam[i].append(frame_keypoints)
            else:
                kpts_cam[i].append(_empty_frame_keypoints(len(HAND_LABELS), point_dim=2))

        #calculate 3d position
        frame_p3ds = []
        
        for hand0_keypoints, hand1_keypoints in zip(kpts_cam[cam_ids[0]][-1], kpts_cam[cam_ids[1]][-1]):
            hand0_rays = _unproject_hand_keypoints(hand0_keypoints, cmtx0, dist0, distortion_model0)
            hand1_rays = _unproject_hand_keypoints(hand1_keypoints, cmtx1, dist1, distortion_model1)

            hand_p3ds = []
            for ray0, ray1 in zip(hand0_rays, hand1_rays):
                if ray0[0] == -1 or ray1[0] == -1:
                    _p3d = [-1, -1, -1]
                else:
                    point_3d = triangulate_rays(
                        tvec0,
                        rmat0 @ ray0,
                        tvec1,
                        rmat1 @ ray1,
                    )
                    _p3d = point_3d.astype(np.float32)
                hand_p3ds.append(_p3d)
            frame_p3ds.append(_filter_hand_points_3d(hand_p3ds))

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((len(HAND_LABELS), NUM_HAND_KEYPOINTS, 3))
        kpts_3d.append(frame_p3ds)

        # Draw the hand annotations on the image.
        frame[cam_ids[0]].flags.writeable = True
        frame[cam_ids[1]].flags.writeable = True
        frame[cam_ids[0]] = cv.cvtColor(frame[cam_ids[0]], cv.COLOR_RGB2BGR)
        frame[cam_ids[1]] = cv.cvtColor(frame[cam_ids[1]], cv.COLOR_RGB2BGR)

        if results[cam_ids[0]].multi_hand_landmarks:
          for i, hand_landmarks in enumerate(results[cam_ids[0]].multi_hand_landmarks):
            if i == 0:
                mp_drawing.draw_landmarks(frame[cam_ids[0]], hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=RED))
            else:
                mp_drawing.draw_landmarks(frame[cam_ids[0]], hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=BLUE))

        if results[cam_ids[1]].multi_hand_landmarks:
          for i, hand_landmarks in enumerate(results[cam_ids[1]].multi_hand_landmarks):
            if i == 0:
                mp_drawing.draw_landmarks(frame[cam_ids[1]], hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=RED))
            else:
                mp_drawing.draw_landmarks(frame[cam_ids[1]], hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=BLUE))

        if visualize:
            cv.imshow('cam1', frame[cam_ids[1]])
            cv.imshow('cam0', frame[cam_ids[0]])

        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.

        frame_idx += 1

    if visualize:
        cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam), np.array(kpts_3d)

def handpose3d(streams, output_path, cam_3d_ids = [1, 4], imu_pts=None, timestamps=None, visualize=False):
    input_streams = streams

    kpts_cam, kpts_3d = run_mp(input_streams, None, None, cam_3d_ids, visualize)
    
    kpts_cam = np.array(kpts_cam)

    # if imu_pts is not None:
    #     for kpts, imu_pts in zip(kpts_3d, imu_pts):
    #         for hand_kpts, imu_pt in zip(kpts, imu_pts):
    #             for kpt in hand_kpts:
    #                 if kpt[0] != -1:
    #                     kpt += imu_pt

    # msg_left_hand, msg_right_hand = construct_3d_hand_keypoints_msg(kpts_3d, timestamps)
    write_3d_hand_keypoints_mcap(kpts_3d, timestamps, 'processed_data/hand_keypoints_3d.mcap')
    for i, kpts_2d in enumerate(kpts_cam):
        write_2d_hand_keypoints_mcap(kpts_2d, timestamps, f"/robot0/sensor/camera{i}/pre/hand_keypoints2d", f'processed_data/hand_keypoints_2d_cam{i}.mcap')
    file_paths = [f'processed_data/hand_keypoints_2d_cam{i}.mcap' for i in range(len(kpts_cam))] + ['processed_data/hand_keypoints_3d.mcap']
    # Use safe_merge_mcaps instead of PyMCAP.merge to avoid corrupting files via raw append
    safe_merge_mcaps(file_paths, output_path)

if __name__ == '__main__':

    input_stream1 = 'media/camera1.mp4'
    input_stream2 = 'media/camera4.mp4'
    mcap_path = '/Users/yangxu/Documents/Code/ff9e3e1189504041b9ce21256925377f.mcap'
    # mcap_path = '/home/yang/Downloads/ff9e3e1189504041b9ce21256925377f.mcap'
    imu_topic = '/robot0/sensor/imu'
    video_topics = ['/robot0/sensor/camera1/compressed', '/robot0/sensor/camera4/compressed']

    msg_imu = read_mcap_protobuf(mcap_path, imu_topic)
    msg_cam = read_mcap_protobuf(mcap_path, video_topics[0])
    timestamps = [msg['header']['timestamp'] for msg in msg_cam]
    msg_imu_synced = msg_time_sync(msg_cam, msg_imu)
    imu_pts = calculate_position_from_imu(msg_imu_synced)
    input_streams = [f'processed_data/camera{i}.mp4' for i in range(6)]
    handpose3d(input_streams, cam_3d_ids = [1, 4], imu_pts=imu_pts, timestamps=timestamps, visualize=True)