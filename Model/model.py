import torch
import torch.nn as nn


class restorationNet(nn.Module):
    def __init__(self):
        super(restorationNet, self).__init__()

        # Encoder: 提取特征
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 输入RGB图像
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Bottleneck: 提取高级特征
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder: 生成去雾图像
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # 输出RGB图像
            nn.Tanh()  # 限制输出在[-1, 1]范围
        )

    def forward(self, x):
        # 前向传播
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    # 测试模型
    model = restorationNet()
    print(model)

    # 随机生成一个输入图像 [batch_size, channels, height, width]
    input_image = torch.randn(1, 3, 256, 256)

    # 前向传播测试
    output_image = model(input_image)
    print("输入图像大小:", input_image.shape)
    print("输出图像大小:", output_image.shape)
