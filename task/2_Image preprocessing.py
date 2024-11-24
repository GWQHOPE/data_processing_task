import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 数据集路径
        :param transform: 图像处理操作
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取文件夹内的所有图片文件
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        加载单张图片并进行处理
        :param idx: 索引
        :return: 处理后的图像
        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # 打开图片并转换为 RGB 格式
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            # 应用图像处理
            image = self.transform(image)
        return image


# 图像预处理操作，包括调整大小、转换为 Tensor、归一化和数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),            # 调整大小为 224x224
    transforms.ToTensor(),                   # 转换为 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到 [-1, 1]
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
])

dataset = CustomDataset(root_dir=r'E:\task1\Model\dataset\raw-60', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 遍历数据并打印预处理后的信息
print("开始加载和预处理数据：")
for batch_idx, batch in enumerate(dataloader):
    print(f"批次 {batch_idx + 1}:")
    print(f"图像维度: {batch.shape}")  # 输出每批次图像的形状
    print(f"第一张图片像素值范围: 最小值 = {batch[0].min().item()}, 最大值 = {batch[0].max().item()}")
print("数据加载和预处理完成！")
