import os
import cv2
import numpy as np
from tqdm import tqdm


# 任务2：首先读取视频进行取帧操作 在本python文件中补充取帧方案的代码

def get_frame_num(total_frames, mode, interval=1, num_segments=1, segment_length=4):
    '''补充取帧方案的代码'''

    '''
    根据不同的模式生成帧编号:
    mode = 'all': 获取所有帧
    mode = 'interval': 等间隔取帧
    mode = 'random': 随机取帧
    mode = 'segment': 等间隔取连续片段
    '''
    frame_list = []

    if mode == 'all':  # 获取所有帧
        frame_list = list(range(total_frames))

    elif mode == 'interval':  # 等间隔取帧
        frame_list = list(range(0, total_frames, interval))

    elif mode == 'random':  # 随机取帧
        # frame_list = sorted(np.random.choice(total_frames, interval, replace=False))
        # interval 参数控制随机帧的数量
        frame_list = sorted(np.random.choice(total_frames, min(interval, total_frames), replace=False))


    elif mode == 'segment':  # 等间隔取连续片段
        segment_interval = total_frames // num_segments
        for i in range(num_segments):
            start_frame = i * segment_interval
            end_frame = start_frame + segment_length
            frame_list.extend(list(range(start_frame, min(end_frame, total_frames))))

    return frame_list

def extract_random_frames(video_path, output_dir, num_frames=1):
    '''
    从视频中提取指定数量的帧并保存
    参数：
    - video_path: 视频文件路径
    - output_dir: 保存帧的输出目录
    - num_frames: 需要提取的帧数
    '''
    cap = cv2.VideoCapture(video_path) # 打开视频文件
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取总帧数

    # 生成指定数量的随机帧编号
    frame_nums = get_frame_num(total_frames, mode='random', interval=num_frames)

    # frame_nums = np.sort(np.random.randint(0, total_frames, total_frames))
    # frame_nums = get_frame_num(total_frames)
    for frame_num in frame_nums:
        # 将视频的帧定位到frame_num
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read() # 读取帧数据
        if ret:
            # 构造保存路径，按视频名生成子目录
            output_path = os.path.join(output_dir, f'{os.path.basename(video_path)}/{frame_num}.png').replace('.mp4','')
            os.makedirs(os.path.join(output_dir, f'{os.path.basename(video_path)}').replace('.mp4',''), exist_ok=True)
            cv2.imwrite(output_path, frame) # 保存帧图像
    cap.release()  # 释放视频资源
def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(len(os.listdir(input_dir)))
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            extract_random_frames(video_path, output_dir, num_frames=num_frames)

if __name__ == "__main__":
    num_frames = 20
    main('./datasets/videos', './datasets/pictures')