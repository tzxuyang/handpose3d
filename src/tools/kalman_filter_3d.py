import numpy as np
from filterpy.kalman import KalmanFilter


class Hand3DKalmanFilter:
    """
    One Kalman filter per 3D hand keypoint.

    State:
        [x, y, z, vx, vy, vz]

    Measurement:
        [x, y, z]
    """

    def __init__(
        self,
        num_points=21,
        process_noise=1e-3,
        measurement_noise=5e-4,
        max_missing_frames=5,
    ):
        self.num_points = num_points
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_missing_frames = max_missing_frames

        self.filters = [None] * num_points
        self.missing_counts = [0] * num_points

    def _create_filter(self, xyz, dt):
        kf = KalmanFilter(dim_x=6, dim_z=3)

        # State:
        # [x, y, z, vx, vy, vz]
        kf.x = np.array([
            xyz[0],
            xyz[1],
            xyz[2],
            0.0,
            0.0,
            0.0,
        ], dtype=float)

        # Constant velocity model
        kf.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=float)

        # We only observe xyz
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Initial covariance
        kf.P *= 1e-2

        # Measurement noise
        # Larger -> smoother, but more lag
        kf.R = np.eye(3) * self.measurement_noise

        # Process noise
        # Larger -> more responsive to fast motion
        kf.Q = np.eye(6) * self.process_noise

        return kf

    def _update_dt(self, kf, dt):
        kf.F[0, 3] = dt
        kf.F[1, 4] = dt
        kf.F[2, 5] = dt

    def update(self, points_3d, dt):
        """
        Args:
            points_3d:
                shape = (21, 3)
                invalid point = [-1, -1, -1]

            dt:
                time difference from previous frame, in seconds

        Returns:
            filtered_points:
                shape = (21, 3)
        """

        points_3d = np.asarray(points_3d, dtype=np.float32)

        filtered_points = np.full_like(
            points_3d,
            -1,
            dtype=np.float32,
        )

        for i, point in enumerate(points_3d):
            valid = not np.all(point == -1)

            # First valid measurement for this keypoint
            if valid and self.filters[i] is None:
                self.filters[i] = self._create_filter(
                    point,
                    dt,
                )

                self.missing_counts[i] = 0
                filtered_points[i] = point
                continue

            kf = self.filters[i]

            # Still never initialized
            if kf is None:
                continue

            self._update_dt(kf, dt)

            # Predict every frame
            kf.predict()

            if valid:
                # Measurement available
                kf.update(point)

                self.missing_counts[i] = 0
                filtered_points[i] = kf.x[:3]

            else:
                # Missing / rejected point
                self.missing_counts[i] += 1

                # Short gaps: use Kalman prediction
                if self.missing_counts[i] <= self.max_missing_frames:
                    filtered_points[i] = kf.x[:3]

                # Too many missing frames -> stop predicting
                else:
                    filtered_points[i] = [-1, -1, -1]

        return filtered_points