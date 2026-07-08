from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import io
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch

app = FastAPI()

# 载入姿态侦测模型（服务器启动时只载入一次，效率比较好）
model_path = "pose_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)
# 载入服装分割模型
seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
]

@app.get("/")
def read_root():
    return {"message": "AI Clothing App Backend 已启动！"}

@app.post("/detect-pose")
async def detect_pose(file: UploadFile = File(...)):
    # 读取上传的图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 转换成 MediaPipe 需要的格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # 进行姿态侦测
    result = detector.detect(mp_image)

    if result.pose_landmarks:
        h, w, _ = image.shape
        for pose_landmarks in result.pose_landmarks:
            for start_idx, end_idx in POSE_CONNECTIONS:
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for landmark in pose_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

    # 把处理好的图片转成可以回传的格式
    _, encoded_image = cv2.imencode(".png", image)
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")

@app.post("/segment-clothes")
async def segment_clothes(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = seg_processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = seg_model(**inputs)

    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image_pil.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    clothing_classes = [3, 4, 5, 6, 7]  # 上衣, 裙子, 裤子, 洋装, 皮带
    mask = np.isin(pred_seg, clothing_classes).astype(np.uint8) * 255

    result = cv2.bitwise_and(image_cv, image_cv, mask=mask)

    _, encoded_image = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")