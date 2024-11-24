import torch.optim
import torch.nn as nn
import os
import argparse
import time
from tqdm import tqdm
# from Model.model import restorationNet
from model import restorationNet
import dataloader
from SSIM import SSIM
import torch
import numpy as np

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restoration_net = restorationNet().to(device)

    restoration_net.apply(initialize_weights)
    train_dataset = dataloader.restoration_loader(config.y_path, config.x_path)
    val_dataset = dataloader.restoration_loader(config.y_path_val, config.x_path_val, mode="val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)

    criterion = []
    criterion.append(SSIM().to(device))
    comput_ssim = SSIM().to(device)  # 验证指标

    restoration_net.train()
    Iters = 1

    indexX = []  # 计数损失曲线用
    indexY = []

    for epoch in range(0, config.num_epochs):
        if epoch <= 500:
            config.lr = 0.0003

        print("now lr == %f" % config.lr)

        print("*" * 80 + f" Epoch {epoch} " + "*" * 80)
        optimizer = torch.optim.AdamW(restoration_net.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=0.02)

        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
        for iteration, (gt, hazy) in loop:
            gt = gt.to(device)
            hazy = hazy.to(device)

            try:
                out = restoration_net(hazy)
                ssim_loss = 1 - criterion[0](out, gt)
                loss = ssim_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                Iters += 1

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

            loop.set_description(f'Epoch [{epoch}/{config.num_epochs}]')
            loop.set_postfix(Loss=loss.item())

        val_ssim = []
        print("start Val!")
        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader), ncols=120)
            for id, (val_y, val_x) in loop_val:
                val_y = val_y.to(device)
                val_x = val_x.to(device)

                val_out = restoration_net(val_x)
                iter_ssim = comput_ssim(val_y, val_out)
                val_ssim.append(iter_ssim.item())

                loop_val.set_description(f'VAL [{id}/{len(val_loader)}]')
                loop_val.set_postfix(ssim=iter_ssim.item())

            indexX.append(epoch)
            now = np.mean(val_ssim)
            if indexY == []:
                indexY.append(now)
                print("First epoch，Save！", 'Now Epoch mean SSIM is:', now)
            else:
                now_max = np.argmax(indexY)
                indexY.append(now)
                print('max epoch %i' % now_max, 'SSIM:', indexY[now_max], 'Now Epoch mean SSIM is:', now)

                if now >= indexY[now_max]:
                    weight_name = 'best.pth'
                    torch.save(restoration_net.state_dict(), config.snapshots_folder + weight_name)
                    print("\033[31msave pth!！！！！！！！！！！！！！！！！！！！！\033[0m")

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)

def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Create directory: ", directory)
    else:
        print(directory, " already exists.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    parser.add_argument('--y_path', type=str, default=r".\dataset\raw-60/")
    parser.add_argument('--x_path', type=str, default=r".\dataset\ref-60/")
    parser.add_argument('--y_path_val', type=str, default=r".\dataset\raw-10/")
    parser.add_argument('--x_path_val', type=str, default=r".\dataset\ref-10/")
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--snapshots_folder', type=str, default="checkpoint/")
    parser.add_argument('--cudaid', type=str, default="0", help="choose cuda device id 0-7.")

    config = parser.parse_args()

    mkdir("trained_model")
    mkdir("trainlog")
    mkdir(config.snapshots_folder)

    s = time.time()
    train(config)
    e = time.time()
    print(str(e - s))
