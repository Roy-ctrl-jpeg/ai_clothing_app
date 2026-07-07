import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "pose_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

image_path = "images (1).jpg"  # 换成你的全身照片
mp_image = mp.Image.create_from_file(image_path)

result = detector.detect(mp_image)
image = cv2.imread(image_path)

# 人体 33 个关键点的标准连接方式（官方 BlazePose 定义）
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 手臂
    (11, 23), (12, 24), (23, 24),                       # 躯干
    (23, 25), (25, 27), (27, 29), (29, 31),             # 左腿
    (24, 26), (26, 28), (28, 30), (30, 32),             # 右腿
    (15, 17), (15, 19), (15, 21),                       # 左手
    (16, 18), (16, 20), (16, 22),                       # 右手
]

if result.pose_landmarks:
    print(f"侦测到 {len(result.pose_landmarks)} 个人的姿态！")
    h, w, _ = image.shape

    for pose_landmarks in result.pose_landmarks:
        # 先画连接线
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 再画关键点
        for landmark in pose_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
else:
    print("没有侦测到人体，请换一张照片试试")

cv2.imshow("Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()