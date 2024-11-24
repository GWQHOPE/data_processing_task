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


# 定义图像处理操作，包括调整大小和转换为 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),            # 调整大小为 224x224
    transforms.ToTensor(),                   # 转换为 Tensor
])

# 初始化数据集
dataset = CustomDataset(root_dir=r'E:\task1\Model\dataset\raw-60', transform=transform)

# 使用 DataLoader 按批次加载数据
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 定义输出文件夹
output_dir = r'E:\task1\Model\dataset\raw-60-save'
os.makedirs(output_dir, exist_ok=True)

# 遍历数据并保存预处理后的图像
print("开始保存数据：")
for batch_idx, batch in enumerate(dataloader):
    # 创建当前批次的保存文件夹
    batch_dir = os.path.join(output_dir, f'batch_{batch_idx + 1}')
    os.makedirs(batch_dir, exist_ok=True)

    # 初始化保存图片的计数
    num_images_saved = 0

    for img_idx, img_tensor in enumerate(batch):
        # 将 Tensor 转换为 PIL 图像
        img = transforms.ToPILImage()(img_tensor)
        # 图片重命名并保存图片
        img_save_path = os.path.join(batch_dir, f'image_{img_idx + 1}.jpg')
        img.save(img_save_path)
        num_images_saved += 1

    # 打印当前批次保存的图片数量和文件夹路径
    print(f"批次 {batch_idx + 1} 保存完成，保存图片数量: {num_images_saved}，文件夹路径: {batch_dir}")

print("所有批次数据保存完成！")
