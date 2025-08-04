import os
import copy
import numpy as np
import h5py
from utils.util import save_rgb_images_to_video
from utils.piper_util import RobotUtil, INIT_QPOS
from utils.gaussian_util import GaussianModel, get_piper_gaussian_at_qpos
from utils.augment_util import *
from utils.data_record_util import DataRecorder
import h5py
from mplib import Pose
from scipy.spatial.transform import Rotation
from utils.trans_util import *
import cv2
from pathlib import Path


def main(demo_Dir, output_path, augment_camera_pose, motionPlanning, visualize, augment_lighting, save_video):
    TableHeightCompensation = -0.05  # 桌子高度补偿
    GraspCompensation = 0.01  # 抓取时夹爪开合的补偿
    OrangeHeightCompensation = torch.tensor([0,0,-0.02]).cuda()  # 橙子高度补偿
    effort_grasp_threshold = -0.8
    effort_place_threshold = -1.0
    bias_Z = 0.06
    num_grasp_step = 15  # 抓取阶段的步数
    grasp_width = 0.075 # 抓取时夹爪闭合距离
    img_qulity_scale = 2 # 渲染图像的质量 [480*img_qulity_scale, 640*img_qulity_scale]
    light_correction_factor = [0.8164251759191522, 0.7991589686451075, 0.7832679011371427]


    data_recorder = DataRecorder(save=True, data_path=output_path, image_shape=[480, 640], save_video=True)

    if not augment_camera_pose:
        camera_1, renderer_1 = get_piper_camera_and_renderer(scale=img_qulity_scale)
    # 创建机器人工具类
    robot_util = RobotUtil(control_hz=60)

    demo_list = sorted(os.listdir(demo_Dir))
    for demo_index, trace_path in enumerate(demo_list):
        # 判断文件是否存在

        try:

            data_recorder.clean()

            demo_file = os.path.join(demo_Dir, trace_path)
            # 打开并读取 JSON 文件

            with h5py.File(demo_file, 'r') as file:
                AllJoints = file['obs']['qpos'][:]
                efforts = file['obs']['effort'][:]

            key_frames = [] # 关键帧列表
            qpos_list = []  # 存储所有的 qpos


            for frame_index in range(1, len(efforts)-1):
                if efforts[frame_index-1][-1] > effort_grasp_threshold and \
                    efforts[frame_index][-1] < effort_grasp_threshold and efforts[frame_index+1][-1] < effort_grasp_threshold:
                    key_frames.append(frame_index)

                elif efforts[frame_index-1][-1] < effort_place_threshold and \
                    efforts[frame_index][-1] > effort_place_threshold and efforts[frame_index+1][-1] > effort_place_threshold:
                    if len(key_frames) < 2:
                        key_frames.append(frame_index)
                    else:
                        key_frames[-1] = frame_index

            if not motionPlanning:
                qpos_array = AllJoints.copy()
                qpos_array[:,-1] = qpos_array[:,-1] / 2 # 将末端关节位置除以2
                qpos_array = np.hstack((qpos_array, qpos_array[:,-1].reshape(-1, 1)))  # 添加一个重复的末尾元素
                qpos_list.extend([row for row in qpos_array])


            # print(key_frames)
            if len(key_frames) != 2:
                print(key_frames)
                print(f"数据 {trace_path} 中没有找到完整的抓取和放置动作，跳过该数据。")
                continue

            if motionPlanning:

                GraspCompensation = 0.0  # 运动规划时不需要夹爪补偿
                grasp_qpos = AllJoints[key_frames[0]]
                grasp_mid_qpos = AllJoints[int(key_frames[0] / 3)]
                place_qpos = AllJoints[key_frames[1]]
                place_mid_qpos = AllJoints[int(key_frames[0] + key_frames[1] / 2)]

                grasp_pose = robot_util.get_ee_pose_at_qpos(grasp_qpos)
                grasp_mid_pose = robot_util.get_ee_pose_at_qpos(grasp_mid_qpos)
                place_pose = robot_util.get_ee_pose_at_qpos(place_qpos)
                place_mid_pose = robot_util.get_ee_pose_at_qpos(place_mid_qpos)


                # 第一段 机械臂末端靠近物体
                # print("第一段 机械臂末端靠近物体")
                poses = [Pose(grasp_mid_pose[:3], grasp_mid_pose[3:]),
                        Pose(grasp_pose[:3]+np.array([0.0, 0.0, bias_Z]), grasp_pose[3:]),
                        Pose(grasp_pose[:3], grasp_pose[3:])]            

                mp_qpos = robot_util.get_Piper_qpos(init_qpos=INIT_QPOS, poses=poses, gripper_action=0)
                # mp_qpos = robot_util.get_trajectory(init_qpos=INIT_QPOS, poses=poses, gripper_action=0)
                for action_index in range(len(mp_qpos)):
                    qpos = mp_qpos[action_index]
                    qpos[-2:] = sum(INIT_QPOS[-2:]) / 2 # 保证 机器人夹爪在运动时不发生变化
                    qpos_list.append(qpos)
                    # print(qpos)

                # 第二段 机械臂抓取物体
                # print("第二段 机械臂抓取物体")
                for grasp_index in range(num_grasp_step):
                    qpos[-2:] = ((grasp_width - sum(INIT_QPOS[-2:])) / num_grasp_step * (grasp_index + 1) + sum(INIT_QPOS[-2:])) / 2
                    qpos_list.append(qpos)
                    # print(qpos)
                
                # 更新抓取关键帧
                key_frames[0] = len(qpos_list)
                
                # 第三段 机械臂末端放置物体
                # print("第三段 机械臂末端放置物体")
                poses = [Pose(grasp_pose[:3]+np.array([0.0, 0.0, bias_Z]), grasp_pose[3:]),
                        Pose(place_mid_pose[:3]+np.array([0.0, 0.0, bias_Z]), place_mid_pose[3:]),
                        # Pose(place_pose[:3]+np.array([0.0, 0.0, bias_Z]), place_pose[3:]),
                        Pose(place_pose[:3], place_pose[3:])]
                mp_qpos = robot_util.get_Piper_qpos(init_qpos=qpos, poses=poses, gripper_action=0)
                for action_index in range(len(mp_qpos)):
                    qpos = mp_qpos[action_index]
                    qpos[-2:] = grasp_width/2 # 保证 机器人夹爪在运动时不发生变化
                    qpos_list.append(qpos)
                    # print(qpos)
                

                # 第四段，夹爪松开
                # print("第四段 夹爪松开")
                for grasp_index in range(num_grasp_step):
                    # qpos = place_qpos.copy()
                    # qpos = np.hstack([qpos, qpos[-1]])  # 添加一个重复的末尾元素
                    qpos[-2:] = ((sum(INIT_QPOS[-2:]) - grasp_width) / num_grasp_step * grasp_index + grasp_width) / 2
                    qpos_list.append(qpos)
                    # print(qpos)

                # 更新放置关键帧
                key_frames[1] = len(qpos_list)
                # 第五段 抓取完毕回到初始位置
                # print("第五段 机械臂末端回到初始位置")
                end_pose = robot_util.get_ee_pose_at_qpos(qpos_list[10])
                poses = [Pose(end_pose[:3], end_pose[3:])]
                mp_qpos = robot_util.get_Piper_qpos(init_qpos=qpos, poses=poses, gripper_action=0)
                for action_index in range(len(mp_qpos)):
                    qpos = mp_qpos[action_index]
                    qpos[-2:] = sum(INIT_QPOS[-2:]) / 2 # 保证 机器人夹爪在运动时不发生变化
                    qpos_list.append(qpos)
                    # print(qpos)


            # 计算橙子和碗的绝对位置
            grasp_qpos = qpos_list[key_frames[0]]
            place_qpos = qpos_list[key_frames[1]]
            grasp_endPose = robot_util.get_ee_pose_at_qpos(grasp_qpos)
            place_endPose = robot_util.get_ee_pose_at_qpos(place_qpos)


            put_gaussian_xyz = place_endPose[:3]
            bowl_gaussian_xyz = put_gaussian_xyz.copy()
            bowl_gaussian_xyz[2] = bowl_gaussian_xyz[2] + TableHeightCompensation # 碗的高度补偿
            orange_gaussian_xyz = grasp_endPose[:3]

            # 加载桌子的 3DGS 模型
            table_gaussian = GaussianModel(sh_degree=3)
            table_gaussian.load_ply('data/k1/scene.ply')

            # 加载碗的 3DGS 模型
            bowl_gaussian = GaussianModel(sh_degree=3)
            # bowl_gaussian.load_ply('data/asset/bowl3.ply')
            bowl_gaussian.load_ply('data/k1/bowl.ply')
            bowl_gaussian._xyz = bowl_gaussian._xyz + torch.from_numpy(bowl_gaussian_xyz).cuda()

            # 加载橙子的 3DGS 模型
            orange_gaussian_ori = GaussianModel(sh_degree=3)
            orange_gaussian_ori.load_ply('data/asset/orange.ply')


            # 第一阶段的 orange 高斯模型
            orange_gaussian_s1 = copy.deepcopy(orange_gaussian_ori)
            orange_gaussian_s1._xyz = orange_gaussian_s1._xyz + torch.from_numpy(orange_gaussian_xyz).cuda() + OrangeHeightCompensation

            # 第三阶段的 orange 高斯模型
            orange_gaussian_s3 = copy.deepcopy(orange_gaussian_ori)
            orange_gaussian_s3._xyz = orange_gaussian_s3._xyz + torch.from_numpy(put_gaussian_xyz).cuda() + OrangeHeightCompensation


            # 设置存储图像的列表
            if save_video:
                image_1_list = []
            # image_h_list = []
            
            # print("---------------------------------------------------------------")
            for i in range(len(qpos_list)):
                # 第一段 机器人靠近抓取物体
                if i < key_frames[0]:
                    qpos = qpos_list[i]
                    qpos[-2:] = qpos[-2:] + GraspCompensation
                    # print("第一段 机器人靠近抓取物体")
                    # print(qpos)

                    # 加载特定 qpos 的机器人高斯模型
                    robot_gaussian = get_piper_gaussian_at_qpos(qpos)


                    gaussian_all = GaussianModel(sh_degree=3)
                    gaussian_all.compose([table_gaussian, bowl_gaussian, orange_gaussian_s1, robot_gaussian])
                    
                # 第二段 机器人抓取并放置
                elif i < key_frames[1]:
                    qpos = qpos_list[i]
                    # print("第二段 机器人抓取并放置")
                    # print(qpos)

                    # 加载特定 qpos 的机器人高斯模型
                    robot_gaussian = get_piper_gaussian_at_qpos(np.array(qpos))

                    # 橙子高斯模型变换到抓取位置
                    orange_gaussian_xyz = robot_util.get_ee_pose_at_qpos(qpos)
                    # orange_gaussian_xyz[2] = orange_gaussian_xyz[2] - put_orange_height  # 橙子放在桌面上
                    orange_gaussian_s2 = copy.deepcopy(orange_gaussian_ori)
                    orange_gaussian_s2._xyz = orange_gaussian_s2._xyz + torch.from_numpy(orange_gaussian_xyz[:3]).cuda() + OrangeHeightCompensation

                    gaussian_all = GaussianModel(sh_degree=3)
                    gaussian_all.compose([table_gaussian, bowl_gaussian, orange_gaussian_s2, robot_gaussian])
                
                # 第三段 机器人放置物体后复位
                elif i < key_frames[1]+210:
                    qpos = qpos_list[i]

                    # 抓取时的补偿
                    qpos[-2:] = qpos[-2:] + GraspCompensation
                    # print("第三段 机器人放置物体后复位")
                    # print(qpos)

                    # 加载特定 qpos 的机器人高斯模型
                    robot_gaussian = get_piper_gaussian_at_qpos(qpos)

                    gaussian_all = GaussianModel(sh_degree=3)
                    gaussian_all.compose([table_gaussian, bowl_gaussian, orange_gaussian_s3, robot_gaussian])

                # 超出放置时间范围 [0 key_frames[1]+210] 的数据丢弃, 确保采集时因为操作不当的机械臂抽搐数据不要
                else:
                    continue
                    
                # 如果进行 光线增强
                if augment_lighting:
                    gaussian_all = augment_lighting_for_scene(gaussian_all, scale_range_min=0.8, scale_range_max=1.0, offset_range_min=-0.3, offset_range_max=0.3)

                # 如果进行 相机视角增强
                if augment_camera_pose:
                    camera_1, renderer_1 = get_piper_augmented_camera_and_renderer(scale=img_qulity_scale, camera_pose_range_scale=0.05)
                
                rgb_1 = renderer_1.render(gaussian_all)

                # 亮度校正
                rgb_1[:,:,0] = rgb_1[:,:,0] * light_correction_factor[0]
                rgb_1[:,:,1] = rgb_1[:,:,1] * light_correction_factor[1]
                rgb_1[:,:,2] = rgb_1[:,:,2] * light_correction_factor[2]
                rgb_1 = (np.clip(rgb_1.detach().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)

                # rgb_1 = cv2.cvtColor(rgb_1, cv2.COLOR_RGB2BGR)

                if save_video:
                    image_1_list.append(rgb_1)

                shift_qpos = qpos[:-1].copy()
                shift_qpos[-1] = shift_qpos[-1] * 2
                # print("shift:", shift_qpos)
                data_recorder.append_step_in_current_episode({
                    'qpos': shift_qpos, 
                    'image': rgb_1
                })

                if visualize:
                    cv2.imshow('camera', rgb_1)
                    cv2.waitKey(10)
            if visualize:
                cv2.destroyAllWindows()
            # 保存当前 episode 的数据
            data_recorder.save_current_episode()
            
            if save_video:
                save_rgb_images_to_video(image_1_list, f'{output_path}/side{demo_index}_table0.mp4')
            print("key_frame: ", key_frames)
            if motionPlanning:
                print(f"第{demo_index+1}个数据已完成运动规划！！！")
            else:
                print(f"第{demo_index+1}个数据已完成重播记录！！！")

        except Exception as e:
            print(f"第{demo_index+1}个数据处理错误 {e}！！！")
            continue
        # break


output_path = 'output/bg_aug'
demo_Dir = '/home/admin123/ssd/Xiangkon/TDGS/data/render_randomization_data'
motionPlanning = True
visualize = False # TO DO 改为并行显示
augment_lighting = False
augment_camera_pose = False
save_video = True

augment_lighting_interval = 10 # 把增强间隔加进去


if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    main(demo_Dir, output_path, augment_camera_pose, motionPlanning, visualize, augment_lighting, save_video)
