import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re # 正则表达式模块 可以用来描述或匹配字符串模式的强大工具


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


def rename_images_in_folder(folder_path):
    """
    将文件夹中的图片统一命名为：路径上倒数第二个文件夹的名字 + '_' + 对应数量.jpg
    :param folder_path: 图片所在文件夹路径
    """
    # 获取倒数第二个文件夹的名字
    parent_folder_name = os.path.basename(os.path.dirname(folder_path))

    # 遍历文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for i, file_name in enumerate(image_files):
        # 构造新的文件名
        new_file_name = f"{parent_folder_name}_{i + 1}.jpg"

        # 构造旧文件路径和新文件路径
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_file_name)

        # 重命名文件
        os.rename(old_path, new_path)
    print("图片重命名完成！")


def save_even_indexed_images(input_folder, output_folder):
    """
    保存排序为偶数的图片到目标文件夹
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 获取排序后的所有图片文件   按 文件名的字典序 排序
    # image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    #
    # for i, file_name in enumerate(image_files):
    #     # 仅保存排序为偶数的图片
    #     if i % 2 == 0:
    #         # 打开图片并保存到目标文件夹
    #         image = Image.open(os.path.join(input_folder, file_name))
    #         image.save(os.path.join(output_folder, file_name))
    # print("偶数排序图片保存完成！")

    # 按数字标号偶数保存
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 遍历图片文件
    for file_name in image_files:
        # 提取文件名中的数字
        match = re.search(r'\d+', file_name)
        if match:
            number = int(match.group())  # 提取数字部分
            if number % 2 == 0:  # 判断数字是否为偶数
                # 保存图片到目标文件夹
                image = Image.open(os.path.join(input_folder, file_name))
                image.save(os.path.join(output_folder, file_name))
    print("数字为偶数的图片保存完成！")


# 定义图像预处理操作，包括调整大小、转换为 Tensor、归一化和数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小为 224x224
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到 [-1, 1]
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
])

# 指定输入和输出路径
input_folder = r"E:\task1\Model\dataset\raw-60"
output_folder = r"E:\task1\Model\dataset\even-images"

# 调用重命名函数
rename_images_in_folder(input_folder)

# 调用保存偶数排序图片函数
save_even_indexed_images(input_folder, output_folder)

# 创建数据集和数据加载器
dataset = CustomDataset(root_dir=input_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 遍历数据并打印预处理后的信息
print("开始加载和预处理数据：")
for batch_idx, batch in enumerate(dataloader):
    print(f"批次 {batch_idx + 1}:")
    print(f"图像维度: {batch.shape}")  # 输出每批次图像的形状
    print(f"第一张图片像素值范围: 最小值 = {batch[0].min().item()}, 最大值 = {batch[0].max().item()}")
print("数据加载和预处理完成！")
