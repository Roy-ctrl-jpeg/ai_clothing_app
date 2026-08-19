# AI Clothing App — Backend

A FastAPI backend that powers an AI-based virtual try-on app. It exposes
endpoints for human pose detection, clothing segmentation, and virtual
garment try-on (via the Replicate IDM-VTON model).

## Features

- **Pose detection** — detects body landmarks in a photo using MediaPipe's
  `PoseLandmarker` and draws the skeleton over the image.
- **Clothing segmentation** — segments clothing regions (upper-clothes,
  skirt, pants, dress, belt) out of a photo using a pretrained SegFormer
  model (`mattmdjaga/segformer_b2_clothes`).
- **Virtual try-on** — sends a model photo and a garment photo to the
  [IDM-VTON](https://replicate.com/cuuupid/idm-vton) model hosted on
  Replicate and returns the generated try-on result.

## Project structure

```
main.py                  FastAPI app with all API endpoints
pose_detection.py         Standalone script: pose detection on a local image
segmentation.py           Standalone script: person/background segmentation
clothes_segmentation.py   Standalone script: clothing segmentation on a local image
read_image.py              Standalone script: face detection with OpenCV Haar cascades
hello.py                  Sanity-check script
pose_landmarker.task       MediaPipe pose landmark model (binary, bundled)
selfie_segmenter.tflite    MediaPipe selfie segmentation model (binary, bundled)
```

The standalone scripts (`pose_detection.py`, `segmentation.py`,
`clothes_segmentation.py`, `read_image.py`) are local experimentation
scripts — they open a display window with OpenCV and are not part of the
FastAPI server. `main.py` is the actual backend service.

## Requirements

- Python 3.10+
- A [Replicate](https://replicate.com/) account and API token, for the
  `/try-on` endpoint

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install fastapi uvicorn python-multipart opencv-python numpy \
            mediapipe transformers torch pillow replicate
```

> The exact `torch` install command may vary depending on whether you want
> CPU-only or GPU (CUDA) support — see the
> [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Set your Replicate API token as an environment variable before starting the
server:

```bash
export REPLICATE_API_TOKEN=your_token_here   # On Windows: set REPLICATE_API_TOKEN=your_token_here
```

## Running the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`, and interactive docs
at `http://localhost:8000/docs`.

## API Endpoints

### `GET /`
Health check. Returns a JSON status message.

### `POST /detect-pose`
Detects body pose landmarks in an uploaded image and returns the image with
the skeleton drawn on top.

- **Body**: `multipart/form-data` with a `file` field (image)
- **Returns**: `image/png`

### `POST /segment-clothes`
Segments clothing regions out of an uploaded image and returns only the
clothing pixels (background removed).

- **Body**: `multipart/form-data` with a `file` field (image)
- **Returns**: `image/png`

### `POST /try-on`
Runs a virtual try-on: puts the garment in `garment_file` onto the person in
`model_file`, using the Replicate IDM-VTON model.

- **Body**: `multipart/form-data` with `model_file` and `garment_file` fields
  (images)
- **Returns**: `image/png`

## Notes

- The pose detection and clothing segmentation models are loaded once at
  server startup for better performance.
- `/try-on` writes uploaded files to temporary local files, calls the
  Replicate API, and deletes the temporary files afterward.
- This backend is paired with a Flutter frontend
  (`clothing_app_frontend`) that uploads photos to these endpoints.
