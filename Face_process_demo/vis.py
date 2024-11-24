import cv2
import numpy as np
image_path = '/content/drive/MyDrive/Face_process_demo/datasets/pictures/000/frames/108.png'
image = cv2.imread(image_path)
landmark_path = '/content/drive/MyDrive/Face_process_demo/datasets/pictures/000/landmarks/108.npy'
landmarks = np.load(landmark_path)[0]
def get_five(ldm81):
    groups = [list(range(36, 42)), list(range(42, 48)), [30], [48], [54]]
    points = []
    for group in groups:
        group_landmarks = [ldm81[i] for i in group]
        mean_point = np.mean(group_landmarks, axis=0)
        points.append(mean_point)
    return np.array(points)  

for landmark in landmarks:
    x, y = int(landmark[0]), int(landmark[1])
    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

cv2.imwrite('/content/drive/MyDrive/Face_process_demo/land.png', image)


blank_image = np.zeros((224, 224, 4), np.uint8)  # 使用4通道图像
blank_image[:, :, 3] = 0
circle_radius = 8
for landmark in landmarks:
    x, y = int(landmark[0]), int(landmark[1])
    cv2.circle(blank_image, (x, y), 3, (0, 255, 0, 255), -1)  # 最后一个参数255表示不透明
cv2.imwrite('/content/drive/MyDrive/Face_process_demo/land2.png', blank_image)
    