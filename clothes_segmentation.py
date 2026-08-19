from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import cv2

# Load the pretrained clothing segmentation model
# (the first run will download the model automatically, which takes a moment)
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Load the image
image_path = "images.jpg"  # replace with your own photo
image = Image.open(image_path).convert("RGB")

# Preprocess and run segmentation
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

# Label lookup table for this model's classes
labels = {
    0: "background", 1: "hat", 2: "sunglasses", 3: "upper-clothes", 4: "skirt",
    5: "pants", 6: "dress", 7: "belt", 8: "left-shoe", 9: "right-shoe",
    10: "face", 11: "left-leg", 12: "right-leg", 13: "left-arm", 14: "right-arm",
    15: "bag", 16: "scarf"
}

# Print which classes were detected in the image
detected = np.unique(pred_seg)
print("Detected regions:")
for d in detected:
    print(f"  {d}: {labels.get(d, 'unknown')}")


clothing_classes = [3, 4, 5, 6, 7]  # upper-clothes, skirt, pants, dress, belt
mask = np.isin(pred_seg, clothing_classes).astype(np.uint8) * 255

image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
result = cv2.bitwise_and(image_cv, image_cv, mask=mask)

cv2.imshow("All Clothing", result)
cv2.waitKey(0)
cv2.destroyAllWindows()