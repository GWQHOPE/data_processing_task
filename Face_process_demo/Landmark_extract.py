from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
import dlib
from imutils import face_utils


import numpy as np
import cv2, torch, os
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image,ImageFont,ImageDraw
import matplotlib.font_manager as fm # to create font
import json
import pandas as pd
import math
from glob import glob
import random
seed_value = 1234 
random.seed(seed_value)


from PIL import Image
def facecrop(ids, data_root, dataset_list, face_detector,face_predictor):
    org_path = os.path.join(data_root, dataset_list[ids])
    frames = glob(org_path+'/*.png')
    save_path = org_path.replace('frames', 'landmarks')
    os.makedirs(save_path,exist_ok=True)
    for cnt_frame in range(len(frames)):
        filename_frame = frames[cnt_frame]
        frame = cv2.imread(filename_frame)
        faces = face_detector(frame, 1)
        if len(faces)==0:
            continue
        face_s_max=-1
        landmarks=[]
        size_list=[]
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0,y0=landmark[:,0].min(),landmark[:,1].min()
            x1,y1=landmark[:,0].max(),landmark[:,1].max()
            face_s=(x1-x0)*(y1-y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]
        land_path = filename_frame.replace('.png', '').replace('frames', 'landmarks')
        # print(land_path)
        np.save(land_path, landmarks)
    return

def reorder_landmark(landmark):
    landmark_add=np.zeros((13,2))
    for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
        landmark_add[idx]=landmark[idx_l]
    landmark[68:]=landmark_add
    return landmark


from glob import glob
from functools import partial
from multiprocessing.pool import Pool
def main():
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = '/content/drive/MyDrive/Face_process_demo/lib/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    data_root = '/content/drive/MyDrive/Face_process_demo/datasets/pictures/000/frames'
    dataset_list = glob(data_root + '*', recursive=True)
    n_sample=len(dataset_list)
    ids = [i for i in range(n_sample)]
    # with Pool(processes=os.cpu_count()-10) as p:
    with Pool(processes=max(1, os.cpu_count() - 10)) as p:  # 修复核心数不足的问题
        with tqdm(total=n_sample) as pbar:
            func = partial(facecrop, data_root=data_root, dataset_list=dataset_list, face_predictor=face_predictor,face_detector=face_detector)
            for v in p.imap_unordered(func, ids):
                pbar.update()
if __name__ == '__main__':
    main()