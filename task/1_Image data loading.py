import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
    # 初始化类，获取文件夹中的图像文件列表，并存储可选的预处理操作
        """
        初始化数据集
        :param root_dir: 数据集路径
        :param transform: 图像处理操作
        """
        self.root_dir = root_dir
        self.transform = transform # 存储预处理操作
        # 获取文件夹内的所有图片文件
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
    # 返回数据集中的样本数量
        """
        返回数据集大小
        """
        return len(self.image_files)  # 返回文件夹内图像文件的数量

    def __getitem__(self, idx):
        """
        加载单张图片并进行处理
        :param idx: 索引
        :return: 处理后的图像
        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])  # 获取图片的路径
        # 打开图片并转换为 RGB 格式
        # 使用PIL.Image打开 图像并转换为RGB格式
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            # 应用图像处理
            image = self.transform(image)
        return image # 返回处理后的图像


# 定义图像处理操作
# transform = transforms.Compose([transforms.ToTensor(),  ])

## 图像数据处理和增强
# 定义处理操作流水线：
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小
    # 将图像从PIL格式转换为PyTorch的张量格式（CxHxW） 范围为 [0, 1]
    transforms.ToTensor(),          # 转换为 Tensor 格式
])

# 初始化数据集
dataset = CustomDataset(root_dir=r'E:\task1\Model\dataset\raw-60', transform=transform)

# 使用 DataLoader 按批次加载数据
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 遍历数据并直接打印批次信息
print("开始加载数据并输出每批次信息：")
for batch_idx, batch in enumerate(dataloader):
    print(f"批次 {batch_idx + 1}:")
    print(f"图像张数: {batch.size(0)}")
    print(f"图像维度: {batch.shape}")
print("数据加载完成！")