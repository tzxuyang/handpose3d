from OneEuroFilter import OneEuroFilter
import numpy as np



class Hand2DOneEuro:
    def __init__(
        self,
        freq=30.0,
        mincutoff=0.5,
        beta=0.15,
        dcutoff=1.0,
        max_missing_frames=10,
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
        self.last_smoothed = np.full((2, 21, 2), -1.0, dtype=float)


        self.missing = np.zeros((2, 21), dtype=int)
        self.max_missing_frames = max_missing_frames

    def reset_hand(self, hand_slot):
        self.filters[hand_slot] = [None for _ in range(21)]
        self.missing[hand_slot] = 0
        self.last_smoothed[hand_slot] = -1.0


    def update(self, frame_keypoints, timestamp_s):
        points = np.asarray(frame_keypoints, dtype=float)

        if points.shape != (2, 21, 2):
            raise ValueError("Expected keypoints shape (2, 21, 2)")
        
        observed = (
            np.isfinite(points).all(axis=-1)
            & (points != -1).all(axis=-1)
        )

        # do not change the original data
        measured_points = np.full(points.shape, -1.0, dtype=float)


        for slot in range(2):
            for point_idx in range(21):
                if not observed[slot, point_idx]:
                    self.missing[slot, point_idx] += 1

                    if self.missing[slot, point_idx] > self.max_missing_frames:
                        self.filters[slot][point_idx] = None

                    continue

                # If the point is observed, reset the missing counter and apply the filter.
                if self.missing[slot, point_idx] > 0:
                    self.filters[slot][point_idx] = None

                self.missing[slot, point_idx] = 0

                if self.filters[slot][point_idx] is None:
                    self.filters[slot][point_idx] = [
                        OneEuroFilter(**self.config),
                        OneEuroFilter(**self.config),
                    ]

                point = points[slot, point_idx]
                filter_x, filter_y = self.filters[slot][point_idx]

                measured_points[slot, point_idx] = [
                    filter_x(float(point[0]), timestamp_s),
                    filter_y(float(point[1]), timestamp_s),
                ]


        # update the last smoothed point by the measured points for display points
        self.last_smoothed[observed] = measured_points[observed]

        # clear last smoothed points for missing points that have been missing for too long
        expired = self.missing > self.max_missing_frames
        self.last_smoothed[expired] = -1.0

        # output measured points and display points (last smoothed points)
        display_points = self.last_smoothed.copy()

        return measured_points, display_points