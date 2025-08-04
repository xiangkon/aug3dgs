# coding=utf-8

"""
眼在手外 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂基坐标系的 旋转矩阵和平移向量

"""

import os
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.trans_util import *
from utils.piper_util import RobotUtil




def GetRT_Base2Gripper(joint_path, valid_flag):
    # 读取 joint 数据
    with open(joint_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    dataList = []
    for line in lines[1:]:
        data = line.strip().split(",")[1:]
        # print(data)
        data = [float(part) for part in data]
        dataList.append(data)
    jointsArray = np.array(dataList)[valid_flag] # 将数据转为 numpy 数组形式并只提取出有效值

    jointsArray[:,4] = jointsArray[:,4] + 0.033 # 补偿piper joint5 未正确调零的偏差（后续正确调零后删除）
    # 初始化机器人
    robot_util = RobotUtil()

    R_Base2GripperList = []
    T_Base2GripperList = []
    for joint in jointsArray:
        joint = np.hstack((joint, np.zeros(2))) # 添加夹爪关节
        robot_util.set_qpos(joint)
        RT_Gripper2Base = pose_to_transformation_matrix(robot_util.get_ee_pose())
        RT_Base2Gripper = np.linalg.inv(RT_Gripper2Base) # eye to hand 此处求逆之后 通过 cv2.calibrateCamera 可直接得到 Camera2Base 的变换矩阵
        R_Base2GripperList.append(RT_Base2Gripper[0:3, 0:3])
        T_Base2GripperList.append(RT_Base2Gripper[0:3, 3])

    return R_Base2GripperList, T_Base2GripperList


def compute_fov(camera_matrix, image_size):
    """
    camera_matrix: 3x3 numpy array (K)
    image_size: (w, h)
    return: (fov_x, fov_y) in degrees
    """
    w, h = image_size
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    fov_x = 2 * np.arctan2(w, 2 * fx)
    fov_y = 2 * np.arctan2(h, 2 * fy)
    return fov_x, fov_y

def func():

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L*objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点
    valid_flag = []     # 存储可用数据
    images_list = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')])

    for image_file in images_list:   #标定好的图片在img_Dir路径下，从0.jpg到x.jpg

        image_file = os.path.join(images_path, image_file)

        if os.path.exists(image_file):

            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            valid_flag.append(ret)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    # 标定,得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    fov_x, fov_y = compute_fov(mtx, (480, 640))
    print(f"水平FOV: {fov_x:.5f}")
    print(f"垂直FOV: {fov_y:.5f}")

    print(size)
    print(valid_flag)

    R_Base2Gripper, T_Base2Gripper = GetRT_Base2Gripper(joint_path, valid_flag)

    R, t = cv2.calibrateHandEye(R_Base2Gripper, T_Base2Gripper, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

    return R,t

if __name__ == '__main__':

    calibration_data_Dir = os.path.join(os.getcwd(), "data/calibration_data")
    images_path = os.path.join(calibration_data_Dir, "collected_photos")
    joint_path = os.path.join(calibration_data_Dir, "robot_joint.txt")

    XX = 7 #标定板的中长度对应的角点的个数
    YY = 5 #标定板的中宽度对应的角点的个数
    L = 0.02   #标定板一格的长度  单位为米

    # 旋转、 平移矩阵
    R_Camera2Base, T_Camera2Base = func()
    save_path = "/home/admin123/ssd/Xiangkon/aug3dgs/data/camera_pose/piper_cam_pose.txt"
    # 变换矩阵
    RT_Camera2Base = np.eye(4)
    RT_Camera2Base[0:3, 0:3] = R_Camera2Base
    RT_Camera2Base[0:3, 3] = T_Camera2Base.flatten()
    print("RT_Camera2Base:\n", RT_Camera2Base)

    with open(save_path, "w") as file:  # 打开文件，准备写入
        for row in RT_Camera2Base:  # 遍历二维数组的每一行
            file.write(" ".join(map(str, row)) + "\n")  # 将每个子数组的元素用空格分隔，然后写入文件

