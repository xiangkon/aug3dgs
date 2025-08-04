import os
import re
import json
import cv2
import numpy as np
from tqdm import tqdm

import h5py
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode


class DataRecorder:
    def __init__(self, save, data_path, start_episode_idx=0, image_shape=[480, 640], image_channel_first=False, save_video=False):
        self.save = save
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.save_video = save_video
        self.image_channel_first = image_channel_first
        self.image_processor = Compose([
            Resize(image_shape, interpolation=InterpolationMode.BICUBIC),
        ])
        self.current_episode = []
        self.episode_idx = start_episode_idx

    def append_step_in_current_episode(self, data_dict):
        # data_dict: {'qpos': qpos, 'image_0': image_0}
        # image: np.uint8, range from 0 to 255
        if not self.save:
            return
        self.current_episode.append(data_dict)

    def save_current_episode(self):
        if not self.save:
            return
        num_step = len(self.current_episode)
        # qpos
        qposes = np.array([step['qpos'] for step in self.current_episode])
        if 'velocity' in self.current_episode[0]:
            velocity = np.array([step['velocity'] for step in self.current_episode])
        if 'effort' in self.current_episode[0]:
            effort = np.array([step['effort'] for step in self.current_episode])

        # image
        images_0 = torch.from_numpy(np.array([step['image'] for step in self.current_episode]))
        if not self.image_channel_first:
            images_0 = images_0.permute(0, 3, 1, 2)
        images_0 = self.image_processor(images_0).permute(0, 2, 3, 1).numpy()

        
        with h5py.File(f'{self.data_path}/{str(self.episode_idx).zfill(6)}.h5', 'w') as h5_file:
            h5_file.create_dataset(name='num_step', data=qposes.shape[0])
            obs_group = h5_file.create_group(name='obs')
            obs_group.create_dataset(name='qpos', data=qposes)
            if 'velocity' in self.current_episode[0]:
                obs_group.create_dataset(name='velocity', data=velocity)
            if 'effort' in self.current_episode[0]:
                obs_group.create_dataset(name='effort', data=effort)
            obs_group.create_dataset(name='image', data=images_0, compression='gzip', compression_opts=4)

        # print(f'save episode {self.episode_idx} in {self.data_path}')

        self.current_episode = []
        self.episode_idx += 1

    def clean(self):
        self.current_episode = []

def sort_episode_folders(root_dir):
    """遍历指定目录下的episode子文件夹并按数字排序"""
    # 用于匹配episodeXX格式的正则表达式
    pattern = re.compile(r'^episode(\d+)$')
    episode_folders = []
    
    try:
        # 遍历根目录下的所有条目
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    # 提取数字部分并转换为整数
                    episode_num = int(match.group(1))
                    episode_folders.append((episode_num, entry.path))
        
        # 按数字部分排序
        episode_folders.sort(key=lambda x: x[0])
        
        # 返回排序后的文件夹路径列表
        return [path for _, path in episode_folders]
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def save_to_hdf5(output_path, root_directorys):

    # 检查输出路径是否存在，如果不存在则创建
    os.makedirs(output_path, exist_ok=True)
    demo_index = 80
    data_recorder = DataRecorder(save=True, data_path=output_path, image_shape=[480, 640], start_episode_idx=demo_index, save_video=True)

    for root_directory in os.listdir(root_directorys):
        sorted_folders = sort_episode_folders(os.path.join(root_directorys, root_directory))
    
        for folder in sorted_folders:
            # print(f"正在处理第 {data_index} 个文件夹: {folder}")
            data_recorder.clean()

            joint_sync_path = os.path.join(folder, 'arm', 'jointState', 'joint_single', 'sync.txt')
            img_sync_path = os.path.join(folder, 'camera', 'color', 'Camera', 'sync.txt')

            if not os.path.exists(joint_sync_path) or not os.path.exists(img_sync_path):
                print(f"Warning: {joint_sync_path} or {img_sync_path} does not exist, skipping this folder.")
                continue

            with open(joint_sync_path, 'r') as f:
                json_filenames = [line.strip() for line in f if line.strip()]
            
            with open(img_sync_path, 'r') as f:
                img_filenames = [line.strip() for line in f if line.strip()]

            for i in range(len(json_filenames)):

                json_path = os.path.join(folder, 'arm', 'jointState', 'joint_single', json_filenames[i])
                img_path = os.path.join(folder, 'camera', 'color', 'Camera', img_filenames[i])
                if not os.path.exists(json_path) or not os.path.exists(img_path):
                    print(f"Warning: {json_path} or {img_path} does not exist, skipping this file.")
                    continue
                
                # 获取关节位置数据
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    joint_position = np.array(json_data['position']) if 'position' in json_data else None
                    velocity = np.array(json_data['velocity']) if 'velocity' in json_data else None
                    effort = np.array(json_data['effort']) if 'effort' in json_data else None
                # 获取图像数据
                image = cv2.imread(img_path)

                data_recorder.append_step_in_current_episode({
                    'qpos': joint_position, 
                    'velocity': velocity,
                    'effort': effort,
                    'image': image
                })
            # 保存当前 episode 的数据
            data_recorder.save_current_episode()

            print(f"Saved {folder} episode {demo_index } data to {output_path}")
            demo_index += 1

if __name__ == "__main__":
    # 指定要遍历的根目录
    root_directorys = "/media/admin123/新加卷/data_randomPosition"  # 当前目录，可替换为实际路径
    
    # print("按数字顺序排列的episode文件夹:")
    sorted_folders = sort_episode_folders(root_directorys)
    output_path = '/home/admin123/ssd/Xiangkon/TDGS/data/randomPosition_200'
    save_to_hdf5(output_path, root_directorys)