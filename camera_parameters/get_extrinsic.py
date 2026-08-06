import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
project_root = os.path.dirname(project_root)
print(project_root)
sys.path.insert(0, project_root)

from utils import quat_to_rot_matrix

qx_0 = -0.17214922222016485
qy_0 = 0.9630417840204724
qz_0 = 0.0005496691056088158
qw_0 = 0.20715903403794017

qx_1 = 0.16638461751841505
qy_1 = 0.9666268034267204
qz_1 = -0.0018388855291070023
qw_1 = -0.19479579166476316

if __name__ == '__main__':
    matrix = quat_to_rot_matrix(qx_0, qy_0, qz_0, qw_0)
    print("Rotation matrix: ")
    for row in matrix:
        print(row)

    matrix = quat_to_rot_matrix(qx_1, qy_1, qz_1, qw_1)
    print("Rotation matrix: ")
    for row in matrix:
        print(row)