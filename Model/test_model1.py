import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
from model import restorationNet
import numpy as np
from PIL import Image
import glob

# 命令行参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default=os.path.join('dataset', 'raw-10'))
parser.add_argument('--outpath', type=str, default='Out')
parser.add_argument('--pretrain_path', type=str, default=os.path.join('checkpoint', 'best.pth'))
args = parser.parse_args()

# 图像去雾处理函数
def restoration(image_path, color_net, device):
    try:
        # 加载图像
        data_restoration = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
        data_restoration = np.asarray(data_restoration) / 255.0  # 归一化到[0, 1]
        data_restoration = torch.from_numpy(data_restoration).float()  # 转为torch张量
        data_restoration = data_restoration.permute(2, 0, 1).unsqueeze(0).to(device)  # 调整维度并移动到设备

        # 前向推理
        with torch.no_grad():
            start = time.time()
            enhanced_image = color_net(data_restoration)
            end_time = time.time() - start
            print(f"Inference time for {os.path.basename(image_path)}: {end_time:.4f}s")

        # 保存结果
        result_path = args.outpath
        os.makedirs(result_path, exist_ok=True)
        save_path = os.path.join(result_path, os.path.basename(image_path))
        torchvision.utils.save_image(enhanced_image, save_path)
        print(f"Enhanced image saved to {save_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# 主函数
if __name__ == '__main__':
    try:
        # 设备设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 加载模型
        color_net = restorationNet()
        color_net = nn.DataParallel(color_net).to(device)
        if os.path.exists(args.pretrain_path):
            color_net.load_state_dict(torch.load(args.pretrain_path, map_location=device))
            print(f"Model loaded from {args.pretrain_path}")
        else:
            raise FileNotFoundError(f"Pretrained model not found at {args.pretrain_path}")

        # 获取输入文件列表
        file_path = args.filepath
        test_list = glob.glob(os.path.join(file_path, "*"))
        if not test_list:
            raise ValueError(f"No images found in the directory: {file_path}")

        # 图像处理
        for image in test_list:
            print(f"Processing: {image}")
            restoration(image, color_net, device)
    except Exception as e:
        print(f"Error in main execution: {e}")
