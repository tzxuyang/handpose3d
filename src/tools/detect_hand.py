import os

import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
HAND_LABELS = ('Left', 'Right')
NUM_HAND_KEYPOINTS = 21


def get_physical_hand_label(mp_label):
    """Media Pipe hypothesis that the input is mirrored. But the DAS ego's input is not mirrored, so we need to swap the labels."""
    if mp_label == "Right":
        return "Left"
    elif mp_label == "Left":
        return "Right"
    return mp_label



def _get_hand_slot(results,frame_shape, previous_wrists):
    """Get the physical hand slot for the detected hand."""
    num_detections = len(
        results.multi_hand_landmarks
    )

    if num_detections == 0:
        return {}

    current_wrists = []
    preferred_slots = []
    scores = []

    # --------------------------------
    # Extract evidence
    # --------------------------------
    for detected_index, hand_landmarks in enumerate(
        results.multi_hand_landmarks
    ):
        wrist = hand_landmarks.landmark[0]

        current_wrists.append(
            np.array([
                wrist.x * frame_shape[1],
                wrist.y * frame_shape[0],
            ], dtype=float)
        )

        preferred_slot = None
        score = 0.0

        if (
            results.multi_handedness
            and detected_index < len(results.multi_handedness)
        ):
            classification = (
                results.multi_handedness[detected_index].classification[0]
            )

            score = classification.score

            physical_label = (
                get_physical_hand_label(classification.label)
            )

            if physical_label in HAND_LABELS:
                preferred_slot = (
                    HAND_LABELS.index(
                        physical_label
                    )
                )

        preferred_slots.append(
            preferred_slot
        )

        scores.append(
            score
        )

    # --------------------------------
    # One-hand case
    # --------------------------------
    if num_detections == 1:

        current_wrist = current_wrists[0]
        preferred_slot = preferred_slots[0]
        handedness_score = scores[0]

        image_diag = float(
            np.hypot(frame_shape[0], frame_shape[1])
        )

        distances = {}

        for slot, previous_wrist in enumerate(previous_wrists):
            if previous_wrist is not None:
                distances[slot] = (
                    np.linalg.norm(current_wrist - previous_wrist)
                    / image_diag
                )

        # Initial parameter
        MAX_STEP = 0.08
        MIN_MARGIN = 0.02
        TRUST_SCORE = 0.90

        # Left、Right hand have distances
        if len(distances) == 2:
            ordered_slots = sorted(
                distances,
                key=distances.get
            )

            nearest_slot = ordered_slots[0]
            other_slot = ordered_slots[1]

            nearest_distance = distances[nearest_slot]
            distance_margin = (
                distances[other_slot] - distances[nearest_slot]
            )

            spatial_clear = (
                nearest_distance <= MAX_STEP
                and distance_margin >= MIN_MARGIN
            )

            label_agrees = (
                preferred_slot is None
                or preferred_slot == nearest_slot
            )

            if spatial_clear:
                if label_agrees or handedness_score < TRUST_SCORE:
                    return {0: nearest_slot}


            return {}

        if len(distances) == 1:
            known_slot = next(iter(distances))

            if distances[known_slot] <= MAX_STEP:
                label_agrees = (
                    preferred_slot is None
                    or preferred_slot == known_slot
                )

                if label_agrees or handedness_score < TRUST_SCORE:
                    return {0: known_slot}

                return {}


        if (
            preferred_slot is not None
            and handedness_score >= TRUST_SCORE
        ):
            return {0: preferred_slot}


        return {}
    
    # --------------------------------
    # Two-hand case
    # --------------------------------
    if num_detections == 2:

        # No full previous state
        if (
            previous_wrists[0] is None
            or previous_wrists[1] is None
        ):
            if (
                preferred_slots[0] is not None
                and preferred_slots[1] is not None
                and preferred_slots[0] != preferred_slots[1]
            ):
                return {
                    0: preferred_slots[0],
                    1: preferred_slots[1],
                }

            #fallback
            return {
                0: 0,
                1: 1,
            }

        for i, hand in enumerate(results.multi_hand_landmarks):
            wrist = hand.landmark[0]

            current_wrists.append(
                np.array([
                    wrist.x * frame_shape[1],
                    wrist.y * frame_shape[0]
                ], dtype=float)
            )

            classification = (
                results.multi_handedness[i]
                .classification[0]
            )

            physical_label = (
                get_physical_hand_label(
                    classification.label
                )
            )

            preferred_slots.append(
                HAND_LABELS.index(
                    physical_label
                )
            )

            scores.append(
                classification.score
            )

        #if temporal history is missing. Use MediaPipe result
        if(previous_wrists[0] is None or previous_wrists[1] is None):
            return {
                0: preferred_slots[0],
                1: preferred_slots[1],
            }


        image_diag = np.hypot(
            frame_shape[0],
            frame_shape[1]
        )


        def temporal_cost(
            detection_idx,
            slot
        ):
            return (
                np.linalg.norm(
                    current_wrists[detection_idx]
                    - previous_wrists[slot]
                )
                / image_diag
            )

        def hand_cost(
            detection_idx,
            slot
        ):
            score = scores[detection_idx]

            if slot == preferred_slots[detection_idx]:
                prob = score
            else:
                prob = 1.0 - score

            return -np.log(
                max(prob, 1e-6)
            )

        WT = 1.0
        WH = 0.3

        #caseA: detection0 -> Left, detection1 -> Right
        cost_a = (
            WT * (
                temporal_cost(0, 0) + temporal_cost(1, 1)
            ) 
            + 
            WH * (
                hand_cost(0, 0) + hand_cost(1, 1)
            )
        )


        #caseB: detection0 -> Right, detection1 -> Left
        cost_b = (
            WT * (
                temporal_cost(0, 1) + temporal_cost(1, 0)
            )
            +
            WH * (hand_cost(0, 1) + hand_cost(1, 0)
            )
        )

        print(
            f"joint cost: "
            f"A={cost_a:.4f}, "
            f"B={cost_b:.4f}"
        )

        if cost_a <= cost_b:
            return {
                0: 0,
                1: 1
            }

        return {
            0: 1,
            1: 0
        }
    
    return {}

