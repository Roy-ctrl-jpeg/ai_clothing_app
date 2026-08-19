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

image_path = "images.jpg"  # replace with your own full-body photo
mp_image = mp.Image.create_from_file(image_path)

result = detector.detect(mp_image)
image = cv2.imread(image_path)

# Standard connections between the 33 body landmarks (official BlazePose definition)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (25, 27), (27, 29), (29, 31),             # left leg
    (24, 26), (26, 28), (28, 30), (30, 32),             # right leg
    (15, 17), (15, 19), (15, 21),                       # left hand
    (16, 18), (16, 20), (16, 22),                       # right hand
]

if result.pose_landmarks:
    print(f"Detected {len(result.pose_landmarks)} person/people!")
    h, w, _ = image.shape

    for pose_landmarks in result.pose_landmarks:
        # Draw the connecting lines first
        for start_idx, end_idx in POSE_CONNECTIONS:
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Then draw the landmarks
        for landmark in pose_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
else:
    print("No person detected, please try a different photo")

cv2.imshow("Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()