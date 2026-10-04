# Model Weights Directory

This directory stores the trained PyTorch model weights (`best.pt`) for the **SafeCityAI** Traffic Violation Detection system.

## Setup Instructions

1. Train the YOLOv5 model using the provided Google Colab notebook (`notebook/SafeCityAI_YOLOv5_Training.ipynb`).
2. After training completes, download the best model weights file `best.pt` from the training output (`runs/train/exp/weights/best.pt`).
3. Place `best.pt` in this directory:

```
model/
├── README.md
└── best.pt
```

## Model Information

- **Architecture**: YOLOv5 (Pretrained backbone: `yolov5s.pt`)
- **Input Resolution**: 640x640 pixels
- **Classes**:
  - `0`: Helmet
  - `1`: NoHelmet
  - `2`: LicensePlate

> **Note**: Both the FastAPI server (`api/server.py`) and inference scripts (`inference/image_inference.py`, `inference/video_inference.py`) expect the model weights at `model/best.pt`.
