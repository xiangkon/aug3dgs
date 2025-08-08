import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
from utils.gaussian_util import GaussianModel, GaussianRenderer, Camera, get_robot_gaussian_at_qpos, transform_gaussian, get_T_for_rotating_around_an_axis, PIPER_FIXED_LINK
from utils.piper_util import Piper
import numpy as np


robot = Piper()


def trans_gaussian_robotPart2ori(path, qpos):
    robot_part_gaussian_origin_list = []
    for i in range(len(robot.links)-1):
        robot_part_gaussian = GaussianModel(sh_degree=3)
        robot_part_gaussian.load_ply(f'{path}/ori/link{i}.ply')
        robot_part_gaussian_origin_list.append(robot_part_gaussian)

    T_all = robot.fkine_all(qpos)
    print(T_all)
    
    
    T_dof = []  # Transformation at current pose
    for i in range(len(robot.links)): # 遍历除 piper_hand_tcp 的各个关节
        print(robot.links[i].name)

        if robot.links[i].name in PIPER_FIXED_LINK:
            print(robot.links[i].name, "跳过！！！")
            continue
        T_dof.append(T_all[i+1])  # skip the base link
        # print(robot.links[i].name)

        # print(T_all[i+1])

    robot_part_gaussian_list = copy.deepcopy(robot_part_gaussian_origin_list)
    os.makedirs(f'{path}/trans', exist_ok=True)

    for i in range(len(robot.links)-1): 

        T = torch.from_numpy(np.linalg.inv(T_dof[i].A)).cuda().to(torch.float32)

        robot_part_gaussian_new = transform_gaussian(robot_part_gaussian_list[i], T)
        robot_part_gaussian_new.save_ply(f'{path}/trans/link{i}.ply')

if __name__ == '__main__':

    path = "data/k1//link2"
    qpos = [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.05, 0.05]
    robot_gaussian = trans_gaussian_robotPart2ori(path, qpos)
    print("trans robot gaussian successfully!!!")