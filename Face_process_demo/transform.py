import torch
from PIL import Image
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from tqdm import tqdm
import os
import cv2

# 图像增强和保存功能
if __name__ == "__main__":
    additional_targets = {}
    base_transform = alb.Compose([
        alb.Resize(224, 224),
        alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets=additional_targets)

    # 定义增强转换，例如灰度化、压缩、色彩偏移等
    augmentations = alb.Compose([
        alb.ToGray(p=0.5),  # 灰度化
        alb.ImageCompression(quality_lower=20, quality_upper=50, p=0.5),  # 图像压缩
        alb.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),  # 色彩偏移
        alb.RandomBrightnessContrast(p=0.5),  # 随机亮度对比度调整
    ])


    image_path = '/content/drive/MyDrive/Face_process_demo/datasets/pictures/000/frames/108.png'
    img = Image.open(image_path)
    img = np.asarray(img)
    tmp_imgs = {"image": img}
    input_tensor = base_transform(**tmp_imgs)
    input_tensor = input_tensor['image']

    '''补充代码,保存tensor为图像'''
    # 保存张量为图像
    output_image = input_tensor.permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式
    output_image = (output_image * 255).astype(np.uint8)  # 反归一化并转换为 uint8 格式
    
    # 保存增强后的图像
    output_path = "/content/drive/MyDrive/Face_process_demo/augmented_image.png"  # 保存路径
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    
    print(f"增强后的图像已保存至 {output_path}")