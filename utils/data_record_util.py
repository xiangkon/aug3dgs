import os
import shutil
import h5py
import numpy as np
import torch
import torchvision
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
    
