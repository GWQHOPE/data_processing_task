import os
import cv2
import numpy as np
from tqdm import tqdm


def get_frame_num(total_frames, mode, interval=1, num_segments=1, segment_length=4):
    """
    根据不同的模式生成帧编号:
    - mode='all': 获取所有帧
    - mode='interval': 等间隔取帧
    - mode='random': 随机取帧
    - mode='segment': 等间隔取连续片段
    """
    frame_list = []

    if mode == 'all':  # 获取所有帧
        frame_list = list(range(total_frames))

    elif mode == 'interval':  # 等间隔取帧
        frame_list = list(range(0, total_frames, interval))

    elif mode == 'random':  # 随机取帧
        frame_list = sorted(np.random.choice(total_frames, min(interval, total_frames), replace=False))

    elif mode == 'segment':  # 等间隔取连续片段
        segment_interval = total_frames // num_segments
        for i in range(num_segments):
            start_frame = i * segment_interval
            end_frame = start_frame + segment_length
            frame_list.extend(list(range(start_frame, min(end_frame, total_frames))))

    return frame_list


def extract_frames(video_path, output_dir, mode, **kwargs):
    """
    从视频中提取帧并保存
    参数：
    - video_path: 视频文件路径
    - output_dir: 保存帧的输出目录
    - mode: 取帧模式 ('all', 'interval', 'random', 'segment')
    - kwargs: 模式相关参数 (interval, num_segments, segment_length)
    """
    cap = cv2.VideoCapture(video_path)  # 打开视频文件
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    print(f"视频: {os.path.basename(video_path)}, 总帧数: {total_frames}")

    # 根据模式获取需要的帧编号
    frame_nums = get_frame_num(total_frames, mode, **kwargs)

    # 根据模式生成子文件夹名称
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if mode == 'all':
        sub_dir = 'all_pictures'
    elif mode == 'interval':
        sub_dir = f"interval_{kwargs['interval']}_pictures"
    elif mode == 'random':
        sub_dir = f"random_{kwargs['interval']}_pictures"
    elif mode == 'segment':
        sub_dir = f"segment_{kwargs['segment_length']}_{kwargs['num_segments']}_pictures"
    else:
        raise ValueError("未知模式")

    # 创建输出目录
    frame_output_dir = os.path.join(output_dir, sub_dir, video_name)
    os.makedirs(frame_output_dir, exist_ok=True)

    # 保存帧
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(frame_output_dir, f"{frame_num}.png")
            cv2.imwrite(output_path, frame)
    cap.release()  # 释放视频资源


def main(input_dir, output_dir):
    """
    主程序，选择模式并处理视频
    """
    print("请选择取帧模式:")
    print("1: 全部帧保存")
    print("2: 等间隔取帧保存")
    print("3: 随机取帧保存")
    print("4: 等间隔取连续片段保存")
    mode_choice = input("输入模式编号: ").strip()

    if mode_choice == '1':
        mode = 'all'
    elif mode_choice == '2':
        mode = 'interval'
        interval = int(input("输入帧间隔: ").strip())
    elif mode_choice == '3':
        mode = 'random'
        interval = int(input("输入随机取帧数量: ").strip())
    elif mode_choice == '4':
        mode = 'segment'
        segment_length = int(input("输入片段帧数: ").strip())
        num_segments = int(input("输入片段数量: ").strip())
    else:
        print("无效模式选择，退出程序")
        return

    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            if mode == 'all':
                extract_frames(video_path, output_dir, mode)
            elif mode == 'interval':
                extract_frames(video_path, output_dir, mode, interval=interval)
            elif mode == 'random':
                extract_frames(video_path, output_dir, mode, interval=interval)
            elif mode == 'segment':
                extract_frames(video_path, output_dir, mode, num_segments=num_segments, segment_length=segment_length)


if __name__ == "__main__":
    input_dir = './datasets/videos'  # 输入视频目录
    output_dir = '../datasets/pictures'  # 输出图片目录
    main(input_dir, output_dir)
