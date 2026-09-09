from OneEuroFilter import OneEuroFilter
import numpy as np



class Hand2DOneEuro:
    def __init__(
        self,
        freq=30.0,
        mincutoff=2.0,
        beta=0.01,
        dcutoff=1.0,
        max_missing_frames=5,
    ):
        # start point
        self.config = dict(
            freq=freq,
            mincutoff=mincutoff,
            beta=beta,
            dcutoff=dcutoff,
        )

        # filters[hand_slot][point_index] = [x_filter, y_filter]
        self.filters = [[None for _ in range(21)] for _ in range(2)]
        self.missing = np.zeros((2, 21), dtype=int)
        self.max_missing_frames = max_missing_frames

    def reset_hand(self, hand_slot):
        self.filters[hand_slot] = [None for _ in range(21)]
        self.missing[hand_slot] = 0

    def update(self, frame_keypoints, timestamp_s):
        points = np.asarray(frame_keypoints, dtype=float)

        if points.shape != (2, 21, 2):
            raise ValueError("Expected keypoints shape (2, 21, 2)")

        # do not change the original data
        smoothed = np.full(points.shape, -1.0, dtype=float)

        for slot in range(2):
            for point_idx in range(21):
                point = points[slot, point_idx]

                valid = (
                    np.all(np.isfinite(point))
                    and np.all(point != -1)
                )

                if not valid:
                    self.missing[slot, point_idx] += 1


                    if self.missing[slot, point_idx] > self.max_missing_frames:
                        self.filters[slot][point_idx] = None

                    continue

                self.missing[slot, point_idx] = 0

                if self.filters[slot][point_idx] is None:
                    self.filters[slot][point_idx] = [
                        OneEuroFilter(**self.config),
                        OneEuroFilter(**self.config),
                    ]

                filter_x, filter_y = self.filters[slot][point_idx]

                smoothed[slot, point_idx] = [
                    filter_x(float(point[0]), timestamp_s),
                    filter_y(float(point[1]), timestamp_s),
                ]

        return smoothed