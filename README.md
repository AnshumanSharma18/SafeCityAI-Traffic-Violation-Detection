# SafeCityAI – Traffic Violation Detection using YOLOv5

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![YOLOv5](https://img.shields.io/badge/Model-YOLOv5s-orange.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)

**SafeCityAI** is an AI-powered real-time object detection system designed for automated traffic-rule enforcement. Utilizing **YOLOv5s transfer learning** trained on a custom traffic dataset, the system detects motorcycle riders wearing protective helmets, identifies traffic rule violators operating without helmets, and detects vehicle license plates for automated processing and enforcement.

---

## Key Features & Capabilities

- **Interactive Streamlit Web App**: Live UI for image upload, interactive confidence thresholding, bounding box rendering, and detection summary reporting.
- **Multi-Class Violation Detection**: Detects helmets, helmet violations, and license plates simultaneously.
- **High Accuracy Transfer Learning**: Fine-tuned on 6,500+ traffic images using YOLOv5s backbone.
- **Production REST API**: High-performance FastAPI server providing `/health` check and `/predict` endpoints.
- **Dual Inference Engine**: Dedicated CLI scripts for single image inference and frame-by-frame video processing.
- **Cross-Platform Compatibility**: Supports Linux, macOS, and Windows environments with automatic PyTorch checkpoint deserialization.
- **Render Cloud Ready**: Fully configured for one-click web deployment on Render (`render.yaml`).

---

## Case Study & Problem Statement

Urban traffic safety violations, specifically non-helmet riding on motorbikes, account for over 60% of severe road casualties in urban areas. Manual traffic enforcement by officers is resource-intensive, limited in physical coverage, and susceptible to oversight.

**SafeCityAI** addresses this challenge by deploying computer vision models to CCTV and traffic surveillance feeds, providing continuous automated monitoring, helmet compliance verification, and license plate detection for traffic law enforcement.

---

## Verified Class Mapping

The model is trained to detect **3 target classes**:

| Class ID | Class Name | Description | Enforcement Status |
| :---: | :---: | :--- | :---: |
| `0` | **Helmet** | Rider wearing protective helmet | Compliant |
| `1` | **LicensePlate** | Vehicle license plate for identification | Tracking / Penalty |
| `2` | **NoHelmet** | Rider operating motorcycle without helmet | Traffic Violation |

---

## Model Training & Validation Performance

The model was trained for **20 epochs** using **YOLOv5s transfer learning** on Google Colab with GPU acceleration.

### Genuine Validation Metrics

| Performance Metric | Validation Result |
| :--- | :---: |
| **Precision** | **87.28%** |
| **Recall** | **79.52%** |
| **mAP@0.5** | **86.48%** |
| **mAP@0.5:0.95** | **53.28%** |

> **Note**: Trained model weights are stored in [`model/best.pt`](file:///c:/Users/ANSHUMAN%20SHARMA.LAPTOP-HFGVTEVM/OneDrive/Desktop/Major%20Project/model/best.pt) (14.4 MB) and are committed for live web deployment.

---

## Live Web App (Streamlit)

The project includes an interactive web interface powered by Streamlit (`app.py`).

### Launching Streamlit Locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Render Cloud Deployment Instructions
1. Connect your GitHub repository to [Render](https://render.com).
2. Create a new **Web Service**.
3. Select **Python Environment**.
4. Configure Build and Start commands:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Alternatively, use the included [`render.yaml`](file:///c:/Users/ANSHUMAN%20SHARMA.LAPTOP-HFGVTEVM/OneDrive/Desktop/Major%20Project/render.yaml) for automated Blueprint deployment.

---

## Dataset Information

The project uses the **Roboflow Traffic Dataset** located at `dataset/roboflow_dataset/`.

- **Total Images**: 6,509 images with matching YOLO `.txt` bounding box annotations.
- **Split Breakdown**:
  - **Train Set**: 5,709 images (9,446 bounding boxes)
  - **Validation Set**: 536 images (868 bounding boxes)
  - **Test Set**: 264 images (433 bounding boxes)
- **Annotation Format**: Standard YOLO normalized format (`<class_id> <x_center> <y_center> <width> <height>`).
- **Configuration**: Standardized in [`dataset/data.yaml`](file:///c:/Users/ANSHUMAN%20SHARMA.LAPTOP-HFGVTEVM/OneDrive/Desktop/Major%20Project/dataset/data.yaml).

---

## Project Structure

```text
.
├── app.py                          # Streamlit web application frontend
├── render.yaml                     # Render cloud deployment blueprint
├── api/
│   ├── __init__.py                 # API package initializer
│   └── server.py                   # FastAPI REST API application
├── inference/
│   ├── __init__.py                 # Inference package initializer
│   ├── image_inference.py          # Single image detection & visualization module
│   └── video_inference.py          # Frame-by-frame video stream detection module
├── model/
│   ├── README.md                   # Model weights documentation
│   └── best.pt               # Trained YOLOv5s model weights (20 epochs)
├── dataset/
│   ├── data.yaml                   # YOLOv5 dataset definition file
│   ├── README.md                   # Dataset layout and annotation guide
│   └── roboflow_dataset/           # Complete Roboflow dataset (train/valid/test)
├── notebook/
│   └── SafeCityAI_YOLOv5_Training.ipynb # Complete executable Google Colab training notebook
├── samples/
│   ├── images/                     # Sample input test images
│   └── videos/                     # Sample input test videos
├── outputs/                        # Saved annotated inference outputs
├── tests/
│   └── test_api.py                 # Pytest test suite for API verification
├── requirements.txt                # Python package dependencies (including Streamlit)
├── Dockerfile                      # Docker image specification
├── .gitignore                      # Git repository ignore rules
└── README.md                       # Project documentation
```

---

## Inference Instructions

### 1. Image Inference
Run object detection on an image from the test dataset:
```bash
python -m inference.image_inference --image dataset/roboflow_dataset/test/images/0073797c-a755-4972-b76b-8ef2b31d44ab___new_IMG_20160315_071740-jpg_jpeg.rf.9b641dadf7e1762fdc96e0279d5e65a7.jpg --conf 0.25
```
Annotated results are automatically saved to `outputs/`.

### 2. Video Inference
Run frame-by-frame object detection on an input video:
```bash
python -m inference.video_inference --video samples/videos/traffic_input.mp4 --conf 0.25
```
Annotated MP4 video is written to `outputs/`.

---

## FastAPI REST API

### Launching the Server
Start the Uvicorn web server:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```
Access interactive OpenAPI Swagger documentation at: `http://localhost:8000/docs`

### Endpoint Specifications

#### 1. Service Health Check (`GET /health`)
- **Request**:
  ```bash
  curl -X GET "http://localhost:8000/health"
  ```
- **Response**:
  ```json
  {
    "status": "ok",
    "model_exists": true,
    "model_loaded": true,
    "model_path": "model/best.pt",
    "supported_classes": ["Helmet", "LicensePlate", "NoHelmet"]
  }
  ```

#### 2. Image Detection Endpoint (`POST /predict`)
- **Request**:
  ```bash
  curl -X POST "http://localhost:8000/predict" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@samples/images/test_traffic.jpg"
  ```
- **Response**:
  ```json
  {
    "detections": [
      {
        "class": "LicensePlate",
        "confidence": 0.8515,
        "box": [
          182,
          402,
          430,
          499
        ]
      }
    ],
    "count": 1
  }
  ```

---

## Running Automated Tests

Run unit tests using Pytest:
```bash
pytest tests/test_api.py
```

---

## Docker Deployment

Build and run the API container:
```bash
# Build image
docker build -t safecityai-api .

# Run container
docker run -d -p 8000:8000 --name safecityai-app safecityai-api
```
