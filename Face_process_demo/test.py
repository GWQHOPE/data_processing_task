import os
import cv2
import numpy as np
from tqdm import tqdm


def get_frame_num(total_frames, mode, interval=1, num_segments=1, segment_length=4):
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
    cap = cv2.VideoCapture(video_path)  # 打开视频文件

    if not cap.isOpened():  # 检查视频是否打开成功
        print(f"无法打开视频文件: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    if total_frames <= 0:  # 检查帧数是否合法
        print(f"视频文件帧数无效: {video_path}")
        cap.release()
        return None

    print(f"视频: {os.path.basename(video_path)}, 总帧数: {total_frames}")

    frame_nums = get_frame_num(total_frames, mode, **kwargs)

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

    frame_output_dir = os.path.join(output_dir, sub_dir, video_name)
    os.makedirs(frame_output_dir, exist_ok=True)

    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(frame_output_dir, f"{frame_num}.png")
            cv2.imwrite(output_path, frame)
    cap.release()
    return total_frames  # 返回总帧数


def main(input_dir, output_dir):
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

    video_list = os.listdir(input_dir)
    total_videos = len([f for f in video_list if f.endswith(".mp4")])
    print(f"检测到 {total_videos} 个视频文件，开始处理...")

    for filename in tqdm(video_list):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            try:
                if mode == 'all':
                    total_frames = extract_frames(video_path, output_dir, mode)
                elif mode == 'interval':
                    total_frames = extract_frames(video_path, output_dir, mode, interval=interval)
                elif mode == 'random':
                    total_frames = extract_frames(video_path, output_dir, mode, interval=interval)
                elif mode == 'segment':
                    total_frames = extract_frames(video_path, output_dir, mode, num_segments=num_segments, segment_length=segment_length)
                if total_frames is None:
                    print(f"处理失败: {video_path}")
                else:
                    print(f"视频 {filename} 处理完成，总帧数: {total_frames}")
            except Exception as e:
                print(f"处理 {video_path} 时发生错误: {e}")


if __name__ == "__main__":
    input_dir = './datasets/videos'  # 输入视频目录
    output_dir = './datasets/pictures'  # 输出图片目录
    main(input_dir, output_dir)
