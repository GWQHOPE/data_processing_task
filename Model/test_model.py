import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
# from Model.model import restorationNet
from model import restorationNet
import numpy as np
from PIL import Image
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default=r'dataset\raw-10/')
parser.add_argument('--outpath', type=str, default=r'Out/')
parser.add_argument('--pretrain_path', type=str, default=r'checkpoint\best.pth')
args = parser.parse_args()


def restoration(image_path, color_net):
    data_restoration = Image.open(image_path)
    data_restoration = (np.asarray(data_restoration) / 255.0)
    data_restoration = torch.from_numpy(data_restoration).float()
    data_restoration = data_restoration.permute(2, 0, 1)
    data_restoration = data_restoration.cuda().unsqueeze(0)

    with torch.no_grad():
        start = time.time()
        enhanced_image= color_net(data_restoration)
    end_time = (time.time() - start)
    print(end_time)

    result_path = args.outpath
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    torchvision.utils.save_image(enhanced_image, os.path.join(result_path, os.path.basename(image_path)))

if __name__ == '__main__':

    with torch.no_grad():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        color_net = restorationNet()
        color_net = nn.DataParallel(color_net)
        color_net = color_net.cuda()
        color_net.load_state_dict(torch.load(args.pretrain_path))

        # path setting
        filePath = args.filepath
        test_list = glob.glob(filePath + "/*")

        # inference
        for image in test_list:
            print(image)
            restoration(image, color_net)
