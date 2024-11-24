import numpy as np
from lib.ct.detection import FaceDetector # 从 lib.ct.detection 模块导入，可能是自定义的人脸检测器
import cv2
from lib.utils import flatten, partition # 从 lib.utils 中导入，可能是辅助函数
from tqdm import tqdm
import custom_data_list # 自定
import os
import face_utils

# 设置CUDA环境变量，默认为0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 初始化人脸检测器
detector = FaceDetector(0)

# 将数组划分为指定段的辅助函数
def split_array(array, segment_size):
    segmented_array = []
    for i in range(0, len(array), segment_size):
        segment = array[i:i+segment_size]
        segmented_array.append(segment)
    return segmented_array

# 处理函数：读取帧、检测人脸、裁剪并保存
def process(root):
    # 获取所有图像文件
    data_list = custom_data_list.get_image_files(root)
    print(f"目录 {root} 中找到 {len(data_list)} 个图像文件")
    
    # 排序和分割数据
    data_list = sorted(data_list)
    data_list_split = split_array(data_list, 1)
    image_size = 224

    for clip in tqdm(data_list_split, desc="处理每个图像文件"):
        frames = []
        for frame_name in clip:
            print(f"读取图像: {frame_name}")
            frame = cv2.imread(os.path.join(frame_name))
            if frame is None:
                print(f"无法读取图像: {frame_name}")
                continue
            frames.append(frame)

        if len(frames) > 0:
            # 检测人脸
            detect_res = flatten(
                [detector.detect(item) for item in partition(frames, 1)]
            )
            print(f"检测结果: {detect_res}")

            # 过滤有效的人脸检测结果
            detect_res = get_valid_faces(detect_res, thres=0.5)
            for faces, frame, frame_name in zip(detect_res, frames, clip):
                if len(faces) > 0:
                    bbox, lm5, score = faces[0]
                    frame, landmark, bbox = face_utils.crop_aligned(
                        frame, lm5, landmarks_68=None, bboxes=bbox, aligned_image_size=image_size
                    )
                    bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                    frame_cropped = crop_face_sbi(frame, bbox=bbox, margin=False)
                    frame_cropped = cv2.resize(frame_cropped, (224, 224))

                    # 修改保存路径并创建目录
                    frame_name = frame_name.replace('/frames_ori/', '/frames/')
                    directory_path = os.path.dirname(frame_name)
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                    try:
                        # 保存裁剪后的人脸图像
                        cv2.imwrite(frame_name, frame_cropped)
                    except Exception as e:
                        print(f"保存图像时出错: {frame_name}, 错误信息: {str(e)}")

# 筛选有效人脸的函数
def get_valid_faces(detect_results, max_count=10, thres=0.5, at_least=False):
    new_results = []
    for i, faces in enumerate(detect_results):
        if len(faces) > max_count:
            faces = faces[:max_count]
        l = []
        for j, face in enumerate(faces):
            if face[-1] < thres and not (j == 0 and at_least):
                continue
            box, lm, score = face
            box = box.astype(np.float32)  # 修正数据类型为 float32
            lm = lm.astype(np.float32)
            l.append((box, lm, score))
        new_results.append(l)
    return new_results

# 裁剪人脸的函数
def crop_face_sbi(img, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False, phase='train'):
    assert phase in ['train', 'val', 'test']
    H, W = img.shape[:2]  # 获取图像高度和宽度
    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4
        w1_margin = w / 4
        h0_margin = h / 4
        h1_margin = h / 4
    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand() * 0.6 + 0.2)
        w1_margin *= (np.random.rand() * 0.6 + 0.2)
        h0_margin *= (np.random.rand() * 0.6 + 0.2)
        h1_margin *= (np.random.rand() * 0.6 + 0.2)
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    return img_cropped

# 主程序入口
if __name__ == '__main__':
    roots = [
        '/content/drive/MyDrive/Face_process_demo/datasets/pictures/000/frames_ori',
        '/content/drive/MyDrive/datasets/pictures/002/frames_ori',
        '/content/drive/MyDrive/datasets/pictures/004/frames_ori',
        '/content/drive/MyDrive/datasets/pictures/006/frames_ori',
    ]
    for root in roots:
        print(f"开始处理目录: {root}")
        process(root)
