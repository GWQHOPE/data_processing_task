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
        :return: 处理后的图像和原始文件名
        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # 打开图片并转换为 RGB 格式
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            # 应用图像处理
            image = self.transform(image)
        return image, self.image_files[idx], img_path  # 返回原始路径


# 定义图像处理操作，包括调整大小和转换为 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小为 224x224
    transforms.ToTensor(),         # 转换为 Tensor
])

# 数据集路径
root_dir = r'E:\task1\Model\dataset\raw-60-rename'
# 初始化数据集
dataset = CustomDataset(root_dir=root_dir, transform=transform)

# 使用 DataLoader 按批次加载数据
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 定义输出文件夹
output_dir = r'E:\task1\Model\dataset\raw-60-save'
os.makedirs(output_dir, exist_ok=True)

# 获取倒数第二层文件夹名称作为命名前缀
folder_name = os.path.basename(os.path.dirname(root_dir))

# 创建保存偶数编号图片的文件夹
even_output_dir = os.path.join(output_dir, "even_images")
os.makedirs(even_output_dir, exist_ok=True)

# 遍历数据并处理
print("开始处理和保存数据：")
for batch_idx, batch in enumerate(dataloader):
    # 获取当前批次的图像、文件名和原始路径
    images, file_names, img_paths = batch

    # 遍历每张图片
    for img_idx, (img_tensor, file_name, img_path) in enumerate(zip(images, file_names, img_paths)):
        # 编号从 1 开始
        global_idx = batch_idx * 16 + img_idx + 1

        # 构建新的文件名：<文件夹名>_<编号>.jpg
        new_file_name = f"{folder_name}_{global_idx}.jpg"

        # 将 Tensor 转换为 PIL 图像
        img = transforms.ToPILImage()(img_tensor)

        # 保存偶数编号的图片
        if global_idx % 2 == 0:
            save_path = os.path.join(even_output_dir, new_file_name)
            img.save(save_path)
            print(f"保存图片: {save_path}")

        # 打印所有图片的新名称
        print(f"已处理图片: {file_name} -> {new_file_name}")

        # 重命名原始图片并保存
        new_img_path = os.path.join(root_dir, new_file_name)  # 获取新的路径
        os.rename(img_path, new_img_path)  # 重命名文件
        print(f"重命名图片: {img_path} -> {new_img_path}")

print("所有数据处理完成，偶数编号图片保存完成，原始文件已重命名！")
