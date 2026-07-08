from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import cv2

# 载入预训练的服装分割模型（第一次运行会自动下载模型，需要一点时间）
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# 读取图片
image_path = "images.jpg"  # 换成你的照片
image = Image.open(image_path).convert("RGB")

# 预处理并进行分割
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits.cpu()
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # (height, width)
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

# 这个模型的类别对照表
labels = {
    0: "背景", 1: "帽子", 2: "太阳眼镜", 3: "上衣", 4: "裙子",
    5: "裤子", 6: "洋装", 7: "皮带", 8: "左鞋", 9: "右鞋",
    10: "脸", 11: "左腿", 12: "右腿", 13: "左手", 14: "右手",
    15: "包包", 16: "围巾"
}

# 印出图片中侦测到哪些类别
detected = np.unique(pred_seg)
print("侦测到的部位：")
for d in detected:
    print(f"  {d}: {labels.get(d, '未知')}")


clothing_classes = [3, 4, 5, 6, 7]  # 上衣, 裙子, 裤子, 洋装, 皮带
mask = np.isin(pred_seg, clothing_classes).astype(np.uint8) * 255

image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
result = cv2.bitwise_and(image_cv, image_cv, mask=mask)

cv2.imshow("All Clothing", result)
cv2.waitKey(0)
cv2.destroyAllWindows()