import copy

import numpy as np

def msg_time_sync(ref_msg, target_msg):
    """
    Synchronize the target message to the reference message based on timestamps.
    Args:
        ref_msg: The reference message with a timestamp.
        target_msg: The target message to be synchronized.
    Returns:
        The synchronized target message.
    """
    output_msg = []
    target_index = 0

    for msg in ref_msg:
        ref_time = msg['header']['timestamp']

        for idx in range(target_index, len(target_msg)):
            target_time = target_msg[idx]['header']['timestamp']
            # If the target time is greater than the reference time, break the loop
            if target_time >= ref_time:
                target_index = idx  # Update the target index for the next iteration
                break
            elif idx == len(target_msg) - 1:
                target_index = idx  # Update the target index for the next iteration
                break
            
        # If the target time is less than or equal to the reference time, update the output message
        output_msg.append(target_msg[target_index])
        
    return output_msg  

def quat_to_rot_matrix(qx, qy, qz, qw):
    # Assumes qx, qy, qz, qw is normalized
    return [
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)]
    ]

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

#direct linear transform
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def read_camera_parameters(camera_id):

    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):

    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    with open(filename, 'w') as fout:
        for frame_kpts in kpts:
            frame_kpts = np.asarray(frame_kpts)
            frame_kpts = frame_kpts.reshape((-1, frame_kpts.shape[-1]))
            for kpt in frame_kpts:
                fout.write(' '.join(str(value) for value in kpt) + ' ')
            fout.write('\n')

def write_instrinsic_parameters(file_path, K, dist):
    with open(file_path, 'w') as f:
        f.write("intrinsic:\n")
        for i in range(len(K)):
            if i % 3 == 0:
                f.write(' '.join(str(parameter) for parameter in K[i:i+3]))
                f.write('\n')
        f.write("distortion:\n")
        f.write(' '.join(str(value) for value in dist) + '\n')

def write_extrinsic_parameters(file_path, R, T):
    with open(file_path, 'w') as f:
        f.write("R:\n")
        for row in R:
            f.write(' '.join(str(parameter) for parameter in row))
            f.write('\n')
        f.write("T:\n")
        for transpose in T:
            f.write(str(transpose) + '\n')

if __name__ == '__main__':

    P2 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
