import torch.nn as nn
import cv2
import torch

# 计算直方图
def comput_hist(img):
   #传入RGB格式的图像（作为numpy数组）
    hist = torch.zeros(256,3)
    for i in range(3):
        # R G B
        # 将numpy数组转换为PyTorch张量
        hist[:,i:i+1] = torch.from_numpy(cv2.calcHist([img], [i], None, [256], [0, 256]))
    return hist

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

# 计算总变量损失
def TV_loss(x):
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)


class tv_loss(nn.Module):
    def __init__(self,tvloss_weight=0.1):
        super(tv_loss, self).__init__()
        self.tvloss_weight = tvloss_weight
    def forward(self,x):
        batch_size = x.shape[0]
        return  self.tvloss_weight*TV_loss(x)/batch_size

if __name__ == "__main__":
    print("")