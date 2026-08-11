import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from extract_mcap import read_mcap_protobuf
from extract_mcap import read_mcap_protobuf
from imu_calculation import calculate_position_from_imu
from utils import DLT, get_projection_matrix, msg_time_sync, write_keypoints_to_disk

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


def run_mp(input_stream1, input_stream2, P0, P1):
    #input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    #set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    #create hand keypoints detector object.
    hands0 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=2, min_tracking_confidence=0.5)
    hands1 = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=2, min_tracking_confidence=0.5)

    #containers for detected keypoints for each camera
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    while True:

        #read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1: break

        #crop to 720x720.
        #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = hands0.process(frame0)
        results1 = hands1.process(frame1)

        #prepare list of hand keypoints of this frame
        #frame0 kpts
        frame0_keypoints = _extract_frame_keypoints(results0, frame0, point_dim=2)
        kpts_cam0.append(frame0_keypoints)

        #frame1 kpts
        frame1_keypoints = _extract_frame_keypoints(results1, frame1, point_dim=2)
        #update keypoints container
        kpts_cam1.append(frame1_keypoints)


        #calculate 3d position
        frame_p3ds = []
        for hand0_keypoints, hand1_keypoints in zip(frame0_keypoints, frame1_keypoints):
            hand_p3ds = []
            for uv1, uv2 in zip(hand0_keypoints, hand1_keypoints):
                if uv1[0] == -1 or uv2[0] == -1:
                    _p3d = [-1, -1, -1]
                else:
                    _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
                hand_p3ds.append(_p3d)
            frame_p3ds.append(hand_p3ds)

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((len(HAND_LABELS), NUM_HAND_KEYPOINTS, 3))
        kpts_3d.append(frame_p3ds)

        # Draw the hand annotations on the image.
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        if results0.multi_hand_landmarks:
          for i, hand_landmarks in enumerate(results0.multi_hand_landmarks):
            if i == 0:
                mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=RED))
            else:
                mp_drawing.draw_landmarks(frame0, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=BLUE))

        if results1.multi_hand_landmarks:
          for i, hand_landmarks in enumerate(results1.multi_hand_landmarks):
            if i == 0:
                mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=RED))
            else:
                mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=BLUE))
        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

def handpose3d(stream1, stream2, imu_pts=None):
    input_stream1 = stream1
    input_stream2 = stream2
    
    #projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
    
    kpts_cam0, kpts_cam1, kpts_3d = run_mp(input_stream1, input_stream2, P0, P1)

    if imu_pts is not None:
        for kpts, imu_pts in zip(kpts_3d, imu_pts):
            for hand_kpts, imu_pt in zip(kpts, imu_pts):
                for kpt in hand_kpts:
                    if kpt[0] != -1:
                        kpt += imu_pt
                        
    #this will create keypoints file in current working folder
    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)

if __name__ == '__main__':

    input_stream1 = 'media/camera1.mp4'
    input_stream2 = 'media/camera4.mp4'
    mcap_path = '/Users/yangxu/Documents/Code/ff9e3e1189504041b9ce21256925377f.mcap'
    imu_topic = '/robot0/sensor/imu'
    video_topics = ['/robot0/sensor/camera1/camera_info', '/robot0/sensor/camera4/camera_info']

    msg_imu = read_mcap_protobuf(mcap_path, imu_topic)
    msg_cam = read_mcap_protobuf(mcap_path, video_topics[0])
    msg_imu_synced = msg_time_sync(msg_cam, msg_imu)
    imu_pts = calculate_position_from_imu(msg_imu_synced)
    handpose3d(input_stream1, input_stream2, imu_pts)