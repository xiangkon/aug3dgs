import numpy as np
import math
from utils.gaussian_util import GaussianModel, GaussianRenderer, Camera, get_robot_gaussian_at_qpos, transform_gaussian, get_T_for_rotating_around_an_axis, PIPER_FIXED_LINK
import torch

def pose_to_transformation_matrix(pose, w_frist=True):
    # 构造变换矩阵
    transformation_matrix = np.eye(4)  # 初始化为 4x4 单位矩阵

    # 设置平移部分
    transformation_matrix[0:3, 3] = [pose[0], pose[1], pose[2]]

    # 四元数转旋转矩阵部分
    # 假设四元数是 w, x, y, z 的形式
    if w_frist:
        q_w = pose[3]
        q_x = pose[4]
        q_y = pose[5]
        q_z = pose[6]
    else:
        q_w = pose[6]
        q_x = pose[3]
        q_y = pose[4]
        q_z = pose[5]     

    # 根据四元数构造旋转矩阵
    rotation_matrix = np.array([
        [1 - 2*q_y**2 - 2*q_z**2, 2*q_x*q_y - 2*q_z*q_w, 2*q_x*q_z + 2*q_y*q_w],
        [2*q_x*q_y + 2*q_z*q_w, 1 - 2*q_x**2 - 2*q_z**2, 2*q_y*q_z - 2*q_x*q_w],
        [2*q_x*q_z - 2*q_y*q_w, 2*q_y*q_z + 2*q_x*q_w, 1 - 2*q_x**2 - 2*q_y**2]
    ])

    # 将旋转矩阵放到变换矩阵的左上角
    transformation_matrix[0:3, 0:3] = rotation_matrix

    return transformation_matrix

def calculate_transformation_matrix(translate_x, translate_y, translate_z, rotate_x, rotate_y, rotate_z):
    """
    计算变换矩阵
    
    参数:
    translate_x (float): X 轴平移量
    translate_y (float): Y 轴平移量
    translate_z (float): Z 轴平移量
    rotate_x (float): 绕 X 轴旋转角度（度）
    rotate_y (float): 绕 Y 轴旋转角度（度）
    rotate_z (float): 绕 Z 轴旋转角度（度）
    
    返回:
    numpy.ndarray: 4x4 的变换矩阵
    """
    # 将旋转角度从度数转换为弧度
    rotate_x_rad = math.radians(rotate_x)
    rotate_y_rad = math.radians(rotate_y)
    rotate_z_rad = math.radians(rotate_z)
    
    # 计算绕各个轴的旋转矩阵
    # 绕 X 轴旋转矩阵
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(rotate_x_rad), -np.sin(rotate_x_rad)],
        [0, np.sin(rotate_x_rad), np.cos(rotate_x_rad)]
    ])
    
    # 绕 Y 轴旋转矩阵
    rotation_y = np.array([
        [np.cos(rotate_y_rad), 0, np.sin(rotate_y_rad)],
        [0, 1, 0],
        [-np.sin(rotate_y_rad), 0, np.cos(rotate_y_rad)]
    ])
    
    # 绕 Z 轴旋转矩阵
    rotation_z = np.array([
        [np.cos(rotate_z_rad), -np.sin(rotate_z_rad), 0],
        [np.sin(rotate_z_rad), np.cos(rotate_z_rad), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵（绕 Z→Y→X 轴的顺序）
    rotation = rotation_x @ (rotation_y @ rotation_z)
    
    # 构造平移矩阵
    translation = np.array([translate_x, translate_y, translate_z])
    
    # 构造变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix

def transorm_3dgs_objects(obj_path, output_path, trans):
    """
    对 3D 高斯对象进行变换
    
    参数:
    obj_path (str): 高斯对象路径
    output_path (str): 输出路径
    trans (numpy.ndarray): 4x4 的变换矩阵
    """
    obj_gaussian = GaussianModel(sh_degree=3)
    obj_gaussian.load_ply(obj_path)

    trans = torch.from_numpy(np.linalg.inv(trans)).cuda().to(torch.float32)
    transformed_gaussian = transform_gaussian(obj_gaussian, trans)
    transformed_gaussian.save_ply(output_path)

    return 
