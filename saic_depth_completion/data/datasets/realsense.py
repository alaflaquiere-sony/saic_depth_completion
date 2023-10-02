import os
import torch

import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter
from random import randrange
import math


class Realsense:
    def __init__(
            self, split="train", dataset_path="", split_path="",transforms=None
    ):
        self.transforms = transforms
        self.data_root = dataset_path
        self.split_file = os.path.join(split_path, split + "_realsense.txt")
        self.data_list = self._get_data_list(self.split_file)
        self.color_name, self.depth_name, self.render_name = [], [], []

        self._load_data()

    def _load_data(self):
        for x in os.listdir(self.data_root):
            scene               = os.path.join(self.data_root, x)
            raw_depth_scene     = os.path.join(scene, 'undistorted_depth_images')
            raw_color_scene  = os.path.join(scene, 'undistorted_color_images')

            for y in os.listdir(raw_depth_scene):
                raw_depth_f     = os.path.join(raw_depth_scene, y)
                color_f         = os.path.join(raw_color_scene, y)

                self.depth_name.append(raw_depth_f)
                self.color_name.append(color_f)

    def _get_data_list(self, filename):
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        data_list = []
        for ele in content:
            left, _, right = ele.split('/')
            name, png = right.split('.')
            data_list.append((left, name))
        return set(data_list)

    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        color           = np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255.
        depth           = np.array(Image.open(self.depth_name[index]))

        mask = np.zeros_like(depth)
        mask[np.where(depth > 0)] = 1
        depth = depth / 1000.

        return  {
            'raw_color':        torch.tensor(color, dtype=torch.float32),
            'color':        torch.tensor(color, dtype=torch.float32),
            'raw_depth':    torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
            'mask':         torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'mask2':         torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'gt_depth':     torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
        }