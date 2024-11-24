import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from common import comput_hist
import glob
import random
import cv2
random.seed(1143)

def populate_train_list(orig_images_path, input_images_path,):

    image_list_haze_index = os.listdir(input_images_path)
    image_dataset = []
    for i in image_list_haze_index:

        image_dataset.append((orig_images_path + i, input_images_path + i))

    train_list = image_dataset
    return train_list

def populate_val_list(orig_images_path, input_images_path,):


    image_list_haze_index = os.listdir(input_images_path)  # 文件名
    image_dataset = []
    for i in image_list_haze_index:  # 添加路径，并组合为元组
        image_dataset.append((orig_images_path + i, input_images_path + i))

    val_list = image_dataset

    return val_list


class restoration_loader(data.Dataset):

    def __init__(self, orig_images_path, input_images_path, mode='train'):



        if mode == 'train':
            self.train_list = populate_train_list(orig_images_path, input_images_path)
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:

            self.val_list = populate_val_list(orig_images_path, input_images_path)
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_taget_path, data_input_path = self.data_list[index]
        data_taget = np.array(Image.open(data_taget_path).convert('RGB'))
        data_input = np.array(Image.open(data_input_path).convert('RGB'))

        data_taget = cv2.resize(data_taget, (256, 256))
        data_input = cv2.resize(data_input, (256, 256))

        data_input = data_input/ 255.0
        data_taget = data_taget/ 255.0

        data_taget = torch.from_numpy(data_taget).float()
        data_input = torch.from_numpy(data_input).float()
        return data_taget.permute(2, 0, 1), data_input.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
