import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "selfie_segmenter.tflite"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    output_category_mask=True
)
segmenter = vision.ImageSegmenter.create_from_options(options)

image_path = "images.jpg"  # 换成你的照片
mp_image = mp.Image.create_from_file(image_path)

result = segmenter.segment(mp_image)
category_mask = result.category_mask.numpy_view()

image = cv2.imread(image_path)

# category_mask: 0 = 背景, 1 = 人
mask = (category_mask < 0.5).astype(np.uint8) * 255

# 把背景变绿色，方便你看出分割效果
green_bg = np.zeros_like(image)
green_bg[:] = (0, 255, 0)

mask_3ch = cv2.merge([mask, mask, mask])
person_only = np.where(mask_3ch == 255, image, green_bg)

cv2.imshow("Segmentation", person_only)
cv2.waitKey(0)
cv2.destroyAllWindows()