import os
import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random


import cv2

class NyuV2:
    def __init__(
            self, split="train", dataset_path="", split_path="",transforms=None
    ):
        self.split = split
        self.transforms = transforms
        self.data_root = dataset_path
        self.split_file = os.path.join(split_path, split + "_nyu.txt")
        self.data_list = self._get_data_list(self.split_file)
        self.color_name, self.mask_name, self.render_name = [], [], []

        self._load_data()

    def _load_data(self):
        x = self.data_root
        scene               = os.path.join(self.data_root, x)
        mask_folder     = os.path.join(scene, 'mask')
        render_depth_scene  = os.path.join(scene, 'depth')

        for y in os.listdir(render_depth_scene):
            valid, num, png = self._split_matterport_path(y)
            if valid == False:
                continue
            data_id = (num)
            if data_id not in self.data_list:
                continue
            mask_f     = os.path.join(mask_folder, y)
            render_depth_f  = os.path.join(render_depth_scene, y)
            color_f         = os.path.join(scene, 'RGB', y.split('.')[0] + '.jpg')


            self.mask_name.append(mask_f)
            self.render_name.append(render_depth_f)
            self.color_name.append(color_f)

    def _get_data_list(self, filename):
        with open(filename, 'r') as f:
            content = f.read().splitlines()
        data_list = []
        for ele in content:
            valid, num, png = self._split_matterport_path(ele)
            if valid == False:
                print(f'Invalid data_id in datalist: {ele}')
            data_list.append((num))
        return set(data_list)

    def _split_matterport_path(self, path):
        try:
            num, png = path.split('.')
            return True, num, png
        except Exception as e:
            print(e)
            return False, None, None, None, None, None

    def __len__(self):
        return len(self.color_name)

    def __getitem__(self, index):
        color           = np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255.
        render_depth    = np.array(Image.open(self.render_name[index])) / 4000.
        mask           = np.array(Image.open(self.mask_name[index])) / 255

        depth = np.copy(render_depth)
        depth[np.where(mask == 0)] = 0

        color = torch.tensor(color, dtype=torch.float32)
        raw_depth = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        gt_depth = torch.tensor(render_depth, dtype=torch.float32).unsqueeze(0)

        if self.split == 'train':
            # Random Crop
            i, j, h, w = transforms.RandomCrop.get_params(color, output_size=(432, 576))
            color = transforms.functional.crop(color, i, j, h, w)
            raw_depth = transforms.functional.crop(raw_depth, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)
            gt_depth = transforms.functional.crop(gt_depth, i, j, h, w)

            # Random flip
            if random.random() > 0.5:
                color = transforms.functional.hflip(color)
                raw_depth = transforms.functional.hflip(raw_depth)
                mask = transforms.functional.hflip(mask)
                gt_depth = transforms.functional.hflip(gt_depth)

            # Random vertical flipping
            if random.random() > 0.5:
                color = transforms.functional.vflip(color)
                raw_depth = transforms.functional.vflip(raw_depth)
                mask = transforms.functional.vflip(mask)
                gt_depth = transforms.functional.vflip(gt_depth)


        return  {
            'color':        color,
            'raw_color':        color, # Need non-normalized color image for viz
            'raw_depth':    raw_depth,
            'mask':         mask,
            'mask2':         torch.ones(mask.shape), # Need mask of the gt's missing depth
            'gt_depth':     gt_depth,
        }