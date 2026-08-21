**Real time 3D hand pose estimation using MediaPipe**

This is a demo on how to obtain 3D coordinates of hand keypoints using MediaPipe and two calibrated cameras. Two cameras are required as there is no way to obtain 3D coordinates from a single camera. Check here: [stereo calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate) for a calibration package. Also my blog post on how to stereo calibrate two cameras: [link](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html). Alternatively, follow the camera calibration at Opencv documentations: [link](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html). If you want to know some details on how this code works, take a look at my accompanying blog post here: [link](https://temugeb.github.io/python/computer_vision/2021/06/27/handpose3d.html).

**MediaPipe**  
Install mediapipe in your virtual environment using:
```
pip install mediapipe
```

**Requirements**  
```
Mediapipe
Python3.8
Opencv
matplotlib
```

**Usage: Generate 2D & 3D coordinates**  
The ```handpose3d.py``` program creates a 3D coordinates file: ```processed_data/handpoints_3d.mcap```. To view the recorded 3D coordinates, simply call:
```
uv run main.py --mode process --mcap_path /home/yang/Downloadsff9e3e1189504041b9ce21256925377f.mcap
```
**Usage: Visualize 2D & 3D coordinates**  
The ```show_hands.py``` program visualize the 2D coordinates on top of the image, and 3D coordinates in world coordinate
```
uv run main.py --mode visualize
```

**Methodology**
***Calibration
mcap_utils.py extract video raw mp4 files from mcap file, it also read camera info from message such as "/robot0/sensor/camera0/camera_info"
write the intrinsic calibration to c0.dat and extrinsic calibration (rotation R and translation T) to rot_trans_c0.data

***2D handpoint detection
use mediapipe from google to detect 2D handpoint in pixels (x, y), the 2D result are composed as a list of size (2, 21, 2)

***2D handpoint undistortion
The camera is distorted (calibration shall be saved in c0.dat, and c1.dat). The cv2.undistortPoints() function to get undistorted 2D data point

***3D projection
use 2D handpoint detected from two cameras, and the intrinsic and extrinsic parameters from the two cameras (left cam & right cam) to project 2D coordinates into 3D coordinates

