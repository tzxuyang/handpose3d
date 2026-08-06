import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import DLT

NUM_HAND_KEYPOINTS = 21


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


def visualize_3d(p3ds):
    output_dir = Path('figs')
    output_dir.mkdir(exist_ok=True)

    """Apply coordinate rotations to point z axis as up"""
    Rz = np.array(([[0., -1., 0.],
                    [1.,  0., 0.],
                    [0.,  0., 1.]]))
    Rx = np.array(([[1.,  0.,  0.],
                    [0., -1.,  0.],
                    [0.,  0., -1.]]))

    p3ds_rotated = []
    rotation = Rz @ Rx
    for frame in p3ds:
        frame_kpts_rotated = []
        for hand_kpts in frame:
            hand_kpts_rotated = []
            for kpt in hand_kpts:
                hand_kpts_rotated.append(rotation @ kpt)
            frame_kpts_rotated.append(hand_kpts_rotated)
        p3ds_rotated.append(frame_kpts_rotated)

    """this contains 3d points of each frame"""
    p3ds_rotated = np.array(p3ds_rotated)

    """Now visualize in 3D"""
    thumb_f = [[0,1],[1,2],[2,3],[3,4]]
    index_f = [[0,5],[5,6],[6,7],[7,8]]
    middle_f = [[0,9],[9,10],[10,11],[11, 12]]
    ring_f = [[0,13],[13,14],[14,15],[15,16]]
    pinkie_f = [[0,17],[17,18],[18,19],[19,20]]
    fingers = [pinkie_f, ring_f, middle_f, index_f, thumb_f]
    fingers_colors = ['red', 'blue', 'green', 'black', 'orange']

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, frame_hands in enumerate(p3ds_rotated):
        if i%2 == 0: continue #skip every 2nd frame
        for hand_kpts in frame_hands:
            print(hand_kpts)
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

        #draw axes
        ax.plot(xs = [0,5], ys = [0,0], zs = [0,0], linewidth = 2, color = 'red')
        ax.plot(xs = [0,0], ys = [0,5], zs = [0,0], linewidth = 2, color = 'blue')
        ax.plot(xs = [0,0], ys = [0,0], zs = [0,5], linewidth = 2, color = 'black')

        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-5000, 5000)
        ax.set_xlabel('x')
        ax.set_ylim3d(-5000, 5000)
        ax.set_ylabel('y')
        ax.set_zlim3d(-5000, 5000)
        ax.set_zlabel('z')
        ax.elev = 0.2*i
        ax.azim = 0.2*i
        plt.savefig(output_dir / f'fig_{i}.png')
        plt.pause(0.01)
        ax.cla()


if __name__ == '__main__':

    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
