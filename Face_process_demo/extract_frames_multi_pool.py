from functools import partial
from multiprocessing.pool import Pool
import os
import cv2
import numpy as np
from tqdm import tqdm


def get_frame_num(total_frames):
    '''补充取帧方案的代码'''
    frame_list = []
    return frame_list


def extract_random_frames(ids, dataset_list, input_dir, output_dir, num_frames=1):
    video_path = os.path.join(input_dir,dataset_list[ids])
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nums = np.sort(np.random.randint(0, total_frames, total_frames))
    # frame_nums = get_frame_num(total_frames)
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f'{os.path.basename(video_path)}/{frame_num}.png').replace('.mp4','')
            os.makedirs(os.path.join(output_dir, f'{os.path.basename(video_path)}').replace('.mp4',''), exist_ok=True)
            cv2.imwrite(output_path, frame)
    cap.release()



def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(len(os.listdir(input_dir)))
    video_list = os.listdir(input_dir)
    n_sample=len(video_list)
    ids = [i for i in range(n_sample)]
    with Pool(processes=os.cpu_count()-20) as p:
        with tqdm(total=n_sample) as pbar:
            func = partial(extract_random_frames, dataset_list=video_list, input_dir=input_dir,output_dir=output_dir)
            for v in p.imap_unordered(func, ids):
                pbar.update()


if __name__ == "__main__":
    num_frames = 20
    methods = ['danet', 'facevid2vid', 'hyperreenact', 'lia', 'mcnet']
    for method in methods:
        main(f'/data4-16T/liuyingjie/datasets/DF-40/video/{method}/ff/video', \
            f'/data4-16T/liuyingjie/datasets/DF-40/video/{method}/ff/frames_ori')

        
