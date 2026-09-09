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
        TRUST_SCORE = 0.95

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
        MAX_STEP = 0.08
        LABEL_WEIGHT = 0.10
        MIN_MOTION_MARGIN = 0.05
        MIN_COST_MARGIN = 0.05
        INIT_SCORE = 0.95
        INIT_MIN_SEPARATION = 0.02

        image_diag = max(float(np.hypot(*frame_shape[:2])), 1.0)
        wrists = np.asarray(current_wrists, dtype=float)

        if not np.isfinite(wrists).all():
            return {}

        known_slots = [
            slot for slot in range(2)
            if previous_wrists[slot] is not None
        ]

        # 行：detection；列：Left / Right slot。
        distances = np.full((2, 2), np.inf)

        for slot in known_slots:
            distances[:, slot] = (
                np.linalg.norm(
                    wrists - previous_wrists[slot],
                    axis=1,
                )
                / image_diag
            )

        # 历史不完整：需要较明确的标签来初始化。
        # 不再使用 detection 0 -> Left 的兜底。
        if len(known_slots) < 2:
            labels_clear = (
                all(slot is not None for slot in preferred_slots)
                and preferred_slots[0] != preferred_slots[1]
                and all(
                    np.isfinite(s) and s >= INIT_SCORE
                    for s in scores
                )
            )

            separation = (
                np.linalg.norm(wrists[0] - wrists[1]) / image_diag
            )

            if not labels_clear or separation < INIT_MIN_SEPARATION:
                return {}

            assignment = {
                0: preferred_slots[0],
                1: preferred_slots[1],
            }

            # 如果还有一只手的有效历史，初始化不能忽略它。
            for det, slot in assignment.items():
                if slot not in known_slots:
                    continue

                matched = distances[det, slot]
                other = distances[1 - det, slot]

                if matched > MAX_STEP:
                    return {}

                if (other - matched) / MAX_STEP < MIN_MOTION_MARGIN:
                    return {}

            return assignment

        # A：detection 0 -> Left，detection 1 -> Right。
        # B：detection 0 -> Right，detection 1 -> Left。
        permutations = ((0, 1), (1, 0))

        motion_costs = np.full(2, np.inf)
        total_costs = np.full(2, np.inf)

        for idx, slots in enumerate(permutations):
            steps = np.array([
                distances[det, slot]
                for det, slot in enumerate(slots)
            ])

            # 任一匹配移动过远，整个排列不可用。
            if np.any(steps > MAX_STEP):
                continue

            # 可接受匹配的位置代价归一化到 [0, 1]。
            motion_costs[idx] = np.mean(steps / MAX_STEP)

            label_penalties = []

            for det, slot in enumerate(slots):
                if (
                    preferred_slots[det] is None
                    or not np.isfinite(scores[det])
                ):
                    probability = 0.5
                else:
                    score = float(np.clip(scores[det], 0.5, 1.0))
                    probability = (
                        score
                        if slot == preferred_slots[det]
                        else 1.0 - score
                    )

                # 有界代价，避免 -log(prob) 在小概率时放大。
                label_penalties.append(1.0 - probability)

            total_costs[idx] = (
                motion_costs[idx]
                + LABEL_WEIGHT * np.mean(label_penalties)
            )

        feasible = np.isfinite(total_costs)

        if not feasible.any():
            return {}

        best = int(np.argmin(total_costs))

        if feasible.all():
            # 位置本身无法明确区分两个排列。
            if (
                abs(motion_costs[0] - motion_costs[1])
                < MIN_MOTION_MARGIN
            ):
                return {}

            # 综合证据不足以明确区分两个排列。
            if (
                abs(total_costs[0] - total_costs[1])
                < MIN_COST_MARGIN
            ):
                return {}

            # 标签把选择推向与位置证据相反的排列时，先拒绝。
            if best != int(np.argmin(motion_costs)):
                return {}

        return {
            det: slot
            for det, slot in enumerate(permutations[best])
        }
