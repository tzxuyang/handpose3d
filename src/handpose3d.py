import os

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
    rotate_points_around_z,
)
from pymcap import PyMCAP
from pathlib import Path
from tools.kalman_filter_3d import Hand3DKalmanFilter
from show_hands import hand_points_to_mp_landmarks
from mediapipe.framework.formats import landmark_pb2
from tools.detect_hand import _get_hand_slot
from tools.oneEuroFilter import Hand2DOneEuro

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_shape = [1300, 1600]
HAND_LABELS = ('Left', 'Right')
NUM_HAND_KEYPOINTS = 21
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def _empty_frame_keypoints(num_hands, point_dim):
    return [[[-1] * point_dim for _ in range(NUM_HAND_KEYPOINTS)] for _ in range(num_hands)]

def get_physical_hand_label(mp_label):
    """Media Pipe hypothesis that the input is mirrored. But the DAS ego's input is not mirrored, so we need to swap the labels."""
    if mp_label == "Right":
        return "Left"
    elif mp_label == "Left":
        return "Right"
    return mp_label






# def _get_hand_slot(results, detected_index, filled_slots, frame_shape, previous_wrists):
#     """Get the physical hand slot for the detected hand."""
#     hand_landmarks = results.multi_hand_landmarks[detected_index]
#     wrist = hand_landmarks.landmark[0]

#     current_wrist = np.array([
#         wrist.x * frame_shape[1],
#         wrist.y * frame_shape[0]
#     ], dtype=float)

#     #MediaPipe Handedness
#     preferred_slot = None
#     handedness_score = 0.0

#     if results.multi_handedness and detected_index < len(results.multi_handedness):
#         classification = results.multi_handedness[detected_index].classification[0]
#         mp_label = classification.label
#         handedness_score = classification.score
#         physical_label = get_physical_hand_label(mp_label)


#         if physical_label in HAND_LABELS:
#             preferred_slot = HAND_LABELS.index(physical_label)

#             # if preferred_slot not in filled_slots:
#             #     return preferred_slot

#     # for slot in range(len(HAND_LABELS)):
#     #     if slot not in filled_slots:
#     #         return slot
#     available_slots = [slot for slot in range(len(HAND_LABELS)) if slot not in filled_slots]
#     if not available_slots:
#         return None
    
#     if preferred_slot is not None and preferred_slot in available_slots:
#         if handedness_score > 0.85:
#             print("#MediaPipe#")
#             return preferred_slot

    
#     #Temporal matching
#     distances = {}

#     for slot in available_slots:
#         prev_wrist = previous_wrists[slot]

#         if prev_wrist is not None:
#             distances[slot] = np.linalg.norm(current_wrist - prev_wrist)

#     if distances:
#         nearest_slot = min(distances, key=distances.get)
#         print("Temporal matching alert!!!!!!!!!!!!")
#         return nearest_slot

    
#     #Fallback: if it dont have previous wrist
#     if (preferred_slot is not None and preferred_slot not in filled_slots):
#         print(">>>Fallback<<<")
#         return preferred_slot
    
#     return available_slots[0]




# def _get_hand_slot(results,frame_shape, previous_wrists):
#     """Get the physical hand slot for the detected hand."""
#     num_detections = len(
#         results.multi_hand_landmarks
#     )

#     if num_detections == 0:
#         return {}

#     current_wrists = []
#     preferred_slots = []
#     scores = []

#     # --------------------------------
#     # Extract evidence
#     # --------------------------------
#     for detected_index, hand_landmarks in enumerate(
#         results.multi_hand_landmarks
#     ):
#         wrist = hand_landmarks.landmark[0]

#         current_wrists.append(
#             np.array([
#                 wrist.x * frame_shape[1],
#                 wrist.y * frame_shape[0],
#             ], dtype=float)
#         )

#         preferred_slot = None
#         score = 0.0

#         if (
#             results.multi_handedness
#             and detected_index
#                 < len(results.multi_handedness)
#         ):
#             classification = (
#                 results.multi_handedness[
#                     detected_index
#                 ].classification[0]
#             )

#             score = classification.score

#             physical_label = (
#                 get_physical_hand_label(
#                     classification.label
#                 )
#             )

#             if physical_label in HAND_LABELS:
#                 preferred_slot = (
#                     HAND_LABELS.index(
#                         physical_label
#                     )
#                 )

#         preferred_slots.append(
#             preferred_slot
#         )

#         scores.append(
#             score
#         )

#     # --------------------------------
#     # One-hand case
#     # --------------------------------
#     if num_detections == 1:
#         if preferred_slots[0] is not None:
#             return {
#                 0: preferred_slots[0]
#             }

#         distances = {}

#         for slot in range(len(HAND_LABELS)):
#             prev_wrist = previous_wrists[slot]

#             if prev_wrist is not None:
#                 distances[slot] = np.linalg.norm(
#                     current_wrists[0]
#                     - prev_wrist
#                 )

#         if distances:
#             return {
#                 0: min(
#                     distances,
#                     key=distances.get
#                 )
#             }

#         return {0: 0}
    
#     # --------------------------------
#     # Two-hand case
#     # --------------------------------
#     if num_detections == 2:

#         # No full previous state
#         if (
#             previous_wrists[0] is None
#             or previous_wrists[1] is None
#         ):
#             if (
#                 preferred_slots[0] is not None
#                 and preferred_slots[1] is not None
#                 and preferred_slots[0]
#                     != preferred_slots[1]
#             ):
#                 return {
#                     0: preferred_slots[0],
#                     1: preferred_slots[1],
#                 }

#             #fallback
#             return {
#                 0: 0,
#                 1: 1,
#             }

#         for i, hand in enumerate(results.multi_hand_landmarks):
#             wrist = hand.landmark[0]

#             current_wrists.append(
#                 np.array([
#                     wrist.x * frame_shape[1],
#                     wrist.y * frame_shape[0]
#                 ], dtype=float)
#             )

#             classification = (
#                 results.multi_handedness[i]
#                 .classification[0]
#             )

#             physical_label = (
#                 get_physical_hand_label(
#                     classification.label
#                 )
#             )

#             preferred_slots.append(
#                 HAND_LABELS.index(
#                     physical_label
#                 )
#             )

#             scores.append(
#                 classification.score
#             )

#         #if temporal history is missing. Use MediaPipe result
#         if(previous_wrists[0] is None or previous_wrists[1] is None):
#             return {
#                 0: preferred_slots[0],
#                 1: preferred_slots[1],
#             }


#         image_diag = np.hypot(
#             frame_shape[0],
#             frame_shape[1]
#         )


#         def temporal_cost(
#             detection_idx,
#             slot
#         ):
#             return (
#                 np.linalg.norm(
#                     current_wrists[detection_idx]
#                     - previous_wrists[slot]
#                 )
#                 / image_diag
#             )

#         def hand_cost(
#             detection_idx,
#             slot
#         ):
#             score = scores[detection_idx]

#             if slot == preferred_slots[detection_idx]:
#                 prob = score
#             else:
#                 prob = 1.0 - score

#             return -np.log(
#                 max(prob, 1e-6)
#             )

#         WT = 1.0
#         WH = 0.3

#         #caseA: detection0 -> Left, detection1 -> Right
#         cost_a = (
#             WT * (
#                 temporal_cost(0, 0) + temporal_cost(1, 1)
#             ) 
#             + 
#             WH * (
#                 hand_cost(0, 0) + hand_cost(1, 1)
#             )
#         )


#         #caseB: detection0 -> Right, detection1 -> Left

#         cost_b = (
#             WT * (
#                 temporal_cost(0, 1)
#                 +
#                 temporal_cost(1, 0)
#             )
#             +
#             WH * (
#                 hand_cost(0, 1)
#                 +
#                 hand_cost(1, 0)
#             )
#         )

#         print(
#             f"joint cost: "
#             f"A={cost_a:.4f}, "
#             f"B={cost_b:.4f}"
#         )

#         if cost_a <= cost_b:
#             return {
#                 0: 0,
#                 1: 1
#             }

#         return {
#             0: 1,
#             1: 0
#         }







def _extract_frame_keypoints(results, frame, point_dim, previous_wrists):
    frame_keypoints = _empty_frame_keypoints(len(HAND_LABELS), point_dim)

    if not results.multi_hand_landmarks:
        return frame_keypoints

    assignments = _get_hand_slot(results, frame.shape, previous_wrists)

    for detected_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        

        if detected_index not in assignments:
            continue

        hand_slot = assignments[detected_index]

        for p in range(NUM_HAND_KEYPOINTS):
            pxl_x = int(round(frame.shape[1] * hand_landmarks.landmark[p].x))
            pxl_y = int(round(frame.shape[0] * hand_landmarks.landmark[p].y))
            frame_keypoints[hand_slot][p] = [pxl_x, pxl_y]
        
    # for hand_slot in range(len(HAND_LABELS)):
    #     wrist_point = frame_keypoints[hand_slot][0]

    #     if wrist_point[0] != -1:
    #         previous_wrists[hand_slot] = np.array(
    #             wrist_point,
    #             dtype=float
    #         )
    return frame_keypoints


























# def _extract_frame_keypoints(results, frame, point_dim, previous_wrists):
#     frame_keypoints = _empty_frame_keypoints(len(HAND_LABELS), point_dim)

#     if not results.multi_hand_landmarks:
#         return frame_keypoints

#     filled_slots = set()
#     for detected_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
#         hand_slot = _get_hand_slot(results, detected_index, filled_slots, frame.shape, previous_wrists)

#         if hand_slot is None:
#             continue

#         filled_slots.add(hand_slot)
#         for p in range(NUM_HAND_KEYPOINTS):
#             pxl_x = int(round(frame.shape[1] * hand_landmarks.landmark[p].x))
#             pxl_y = int(round(frame.shape[0] * hand_landmarks.landmark[p].y))
#             frame_keypoints[hand_slot][p] = [pxl_x, pxl_y]
        
#     previous_wrists[hand_slot] = np.array(
#         frame_keypoints[hand_slot][0],
#         dtype=float
#     )

#     return frame_keypoints


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

def run_mp(input_streams, P0, P1, cam_ids = [1,4], visualize=False,timestamps=None):
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
    
    # containers for previous wrist positions for temporal matching
    max_wrist_age_frames = 10
    previous_wrist_frames = {
        cam_id: [None, None]
        for cam_id in range(len(input_streams))
    }
    previous_wrists = {
        cam_id: [
            None,   # Left slot
            None,   # Right slot
        ]
        for cam_id in range(len(input_streams))
    }

    #2d one euro filter
    hand_smoothers = [
        Hand2DOneEuro(freq=30.0)
        for _ in input_streams
    ]

    #3d kalman filter
    # Initialize Kalman filters for right and left hands
    right_hand_kalman = Hand3DKalmanFilter(
        num_points=NUM_HAND_KEYPOINTS,
    )

    left_hand_kalman = Hand3DKalmanFilter(
        num_points=NUM_HAND_KEYPOINTS,
    )



    #3d point missing frame cache
    max_missing_3d_frames = 10
    last_valid_3d = np.full(
        (len(HAND_LABELS), NUM_HAND_KEYPOINTS, 3),
        -1.0,
        dtype=float,
    )

    missing_3d = np.zeros(
        (len(HAND_LABELS), NUM_HAND_KEYPOINTS),
        dtype=int,
    )

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
        display_keypoints = []
        measured_keypoints = []

        frame_timestamps = frame_idx / 30.0
        
        #frame0 kpts
        # for i in range(len(input_streams)):
        #     if frame[i] is not None:
        #         frame_keypoints = _extract_frame_keypoints(results[i], frame[i], point_dim=2, previous_wrists=previous_wrists[i])

        #     else:
        #         frame_keypoints = _empty_frame_keypoints(len(HAND_LABELS), point_dim=2)

        #     kpts_cam[i].append(frame_keypoints)
        for i in range(len(input_streams)):
            # before processing the current frame, check if any previous wrist positions have expired
            for slot in range(len(HAND_LABELS)):
                last_seen = previous_wrist_frames[i][slot]

                if (
                    last_seen is None
                    or frame_idx - last_seen > max_wrist_age_frames
                ):
                    previous_wrists[i][slot] = None
                    previous_wrist_frames[i][slot] = None

            if frame[i] is not None and results[i] is not None:
                frame_keypoints = _extract_frame_keypoints(
                    results[i],
                    frame[i],
                    point_dim=2,
                    previous_wrists=previous_wrists[i],
                )
            else:
                frame_keypoints = _empty_frame_keypoints(
                    len(HAND_LABELS),
                    point_dim=2,
                )

            for slot in range(len(HAND_LABELS)):
                wrist = np.asarray(frame_keypoints[slot][0], dtype=float)

                if np.isfinite(wrist).all() and np.all(wrist != -1):
                    previous_wrists[i][slot] = wrist.copy()
                    previous_wrist_frames[i][slot] = frame_idx

            kpts_cam[i].append(frame_keypoints)

            #apply one euro filter
            measured_points, display_points = hand_smoothers[i].update(
                frame_keypoints,
                frame_timestamps,
            )
            display_keypoints.append(display_points)
            measured_keypoints.append(measured_points)
        

        print('display keypoints: ', display_keypoints[cam_ids[0]], display_keypoints[cam_ids[1]])
        print("displaypoint type", np.asarray(display_keypoints).shape)
        print("kpts type : ", np.asarray(kpts_cam).shape)


        #calculate 3d position
        frame_p3ds = []
        
        # for hand0_keypoints, hand1_keypoints in zip(kpts_cam[cam_ids[0]][-1], kpts_cam[cam_ids[1]][-1]):
        for hand0_keypoints, hand1_keypoints in zip(measured_keypoints[cam_ids[0]], measured_keypoints[cam_ids[1]]):
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
                    _p3d = rotate_points_around_z(point_3d.astype(np.float32), 90)
                hand_p3ds.append(_p3d)
            frame_p3ds.append(_filter_hand_points_3d(hand_p3ds))

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        # frame_p3ds = np.array(frame_p3ds).reshape((len(HAND_LABELS), NUM_HAND_KEYPOINTS, 3))
        measured_p3ds = np.asarray(frame_p3ds, dtype=float).reshape(
            len(HAND_LABELS), NUM_HAND_KEYPOINTS, 3
        )

        dt = 1.0 / 30.0
        if timestamps is not None and 0 < frame_idx < len(timestamps):
            timestamp_delta_ns = (
                int(timestamps[frame_idx])
                - int(timestamps[frame_idx - 1])
            )
            measured_dt = timestamp_delta_ns * 1e-9

            if 0.0 < measured_dt <= 0.1:
                dt = measured_dt


        measured_p3ds[0] = left_hand_kalman.update(measured_p3ds[0], dt)
        measured_p3ds[1] = right_hand_kalman.update(measured_p3ds[1], dt)



        observed_3d = (
            np.isfinite(measured_p3ds).all(axis=-1)
            & ~np.all(measured_p3ds == -1.0, axis=-1)
        )

        #update missing frame count and last valid 3d point cache
        missing_3d[observed_3d] = 0
        missing_3d[~observed_3d] += 1

        last_valid_3d[observed_3d] = measured_p3ds[observed_3d]

        expired = missing_3d > max_missing_3d_frames
        last_valid_3d[expired] = -1.0

        display_p3ds = last_valid_3d.copy()

        kpts_3d.append(display_p3ds)





        # Draw the hand annotations on the image.
        frame[cam_ids[0]].flags.writeable = True
        frame[cam_ids[1]].flags.writeable = True
        frame[cam_ids[0]] = cv.cvtColor(frame[cam_ids[0]], cv.COLOR_RGB2BGR)
        frame[cam_ids[1]] = cv.cvtColor(frame[cam_ids[1]], cv.COLOR_RGB2BGR)

        # if results[cam_ids[0]].multi_hand_landmarks:
        #     for i, hand_landmarks in enumerate(results[cam_ids[0]].multi_hand_landmarks):

        #         handedness = results[cam_ids[0]].multi_handedness[i]
        #         classification = handedness.classification[0]

        #         mp_label = classification.label
        #         score = classification.score

        #         physical_label = get_physical_hand_label(mp_label)

        #         if physical_label == "Right":
        #             color = RED
        #         elif physical_label == "Left":
        #             color = BLUE

        #         mp_drawing.draw_landmarks(
        #             frame[cam_ids[0]],
        #             hand_landmarks,
        #             mp_hands.HAND_CONNECTIONS,
        #             mp_drawing.DrawingSpec(color=color),
        #         )

        # for hand_idx, hand_keypoints in enumerate(frame_keypoints):
        #     if hand_idx == 0:
        #         color = BLUE
        #     else:
        #         color = RED


        # if results[cam_ids[1]].multi_hand_landmarks:
        #     for i, hand_landmarks in enumerate(results[cam_ids[1]].multi_hand_landmarks):

        #         handedness = results[cam_ids[1]].multi_handedness[i]
        #         classification = handedness.classification[0]

        #         mp_label = classification.label
        #         score = classification.score
        #         physical_label = get_physical_hand_label(mp_label)

        #         if physical_label == "Right":
        #             color = RED
        #         elif physical_label == "Left":
        #             color = BLUE

        #         mp_drawing.draw_landmarks(
        #             frame[cam_ids[1]],
        #             hand_landmarks,
        #             mp_hands.HAND_CONNECTIONS,
        #             mp_drawing.DrawingSpec(color=color),
                # )
        for cam_id in cam_ids:

            # current_handpoints = kpts_cam[cam_id][-1]
            current_handpoints = display_keypoints[cam_id]

            for hand_idx, hand_keypoints in enumerate(
                current_handpoints
            ):
                hand_keypoints = np.asarray(
                    hand_keypoints,
                    dtype=float
                )

                # 当前 hand 没有检测到
                if np.all(hand_keypoints[:, 0] == -1):
                    continue

                # hand_idx 已经是 _get_hand_slot() 整理后的 slot
                label = HAND_LABELS[hand_idx]

                if label == "Left":
                    color = BLUE
                elif label == "Right":
                    color = RED
                else:
                    continue

                hand_landmarks = hand_points_to_mp_landmarks(
                    hand_keypoints,
                    frame[cam_id].shape
                )

                mp_drawing.draw_landmarks(
                    frame[cam_id],
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color),
                )



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

    kpts_cam, kpts_3d = run_mp(input_streams, None, None, cam_3d_ids, visualize, timestamps)
    
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
    # Delete the individual files after merging
    file_paths = [f'processed_data/hand_keypoints_2d_cam{i}.mcap' for i in range(len(kpts_cam))]
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"Error: {file_path} does not exist.")
        except PermissionError:
            print(f"Error: You do not have permission to delete {file_path}.")

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