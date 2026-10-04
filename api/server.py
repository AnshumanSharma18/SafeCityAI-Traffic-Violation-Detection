"""SafeCityAI - FastAPI Traffic Violation Detection Server.

Provides REST API endpoints for model health checking and image violation detection.
"""

import io
import pathlib
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

# Fix for loading Linux/Colab-trained PyTorch checkpoints on Windows (Python 3.13+)
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath
    if hasattr(pathlib, "_local"):
        pathlib._local.PosixPath = pathlib.WindowsPath

# Define project root and weights path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "model" / "best.pt"

# Class ID mapping preserving Roboflow dataset order
# 0: Helmet, 1: LicensePlate, 2: NoHelmet
CLASS_NAMES = {0: "Helmet", 1: "LicensePlate", 2: "NoHelmet"}

app = FastAPI(
    title="SafeCityAI API",
    description="Traffic Violation Detection API using YOLOv5 Transfer Learning.",
    version="1.0.0"
)

# Global model instance
model = None


def get_model():
    """Lazy-load model instance if weights file exists."""
    global model
    if model is None:
        if not WEIGHTS_PATH.is_file():
            return None
        try:
            model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=str(WEIGHTS_PATH),
                trust_repo=True,
                force_reload=False
            )
            model.eval()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return None
    return model


# Pydantic Response Models
class HealthResponse(BaseModel):
    status: str = Field(...)
    model_exists: bool = Field(...)
    model_loaded: bool = Field(...)
    model_path: str = Field(...)
    supported_classes: List[str] = Field(default=["Helmet", "LicensePlate", "NoHelmet"])


class DetectionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cls_name: str = Field(..., alias="class")
    confidence: float = Field(...)
    box: List[int] = Field(..., description="[xmin, ymin, xmax, ymax]")


class PredictionResponse(BaseModel):
    detections: List[DetectionItem]
    count: int = Field(...)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API service health and model file availability."""
    exists = WEIGHTS_PATH.is_file()
    mdl = get_model() if exists else None
    return HealthResponse(
        status="ok",
        model_exists=exists,
        model_loaded=mdl is not None,
        model_path=str(WEIGHTS_PATH),
        supported_classes=["Helmet", "LicensePlate", "NoHelmet"]
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict(file: UploadFile = File(...)):
    """Upload an image (JPG/JPEG/PNG) to perform YOLOv5 traffic violation detection.

    Returns detected classes (Helmet, LicensePlate, NoHelmet), confidence score, and bounding boxes.
    """
    # 1. Validate model presence
    if not WEIGHTS_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model weights file 'model/best.pt' not found. "
                "Please train the model using Google Colab and place best.pt in the model/ directory."
            )
        )

    mdl = get_model()
    if mdl is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load trained model weights."
        )

    # 2. Validate input file content type
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.content_type}'. Upload a valid image (JPG/JPEG/PNG)."
        )

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file upload: {str(e)}"
        )

    # 3. Perform YOLOv5 inference
    try:
        # Convert PIL image to OpenCV BGR array
        img_np = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = mdl(img_bgr)
        predictions = results.xyxy[0].cpu().numpy()

        detections = []
        for pred in predictions:
            xmin, ymin, xmax, ymax, conf, cls_id = pred
            cls_id = int(cls_id)
            cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")

            detections.append(
                DetectionItem(
                    **{
                        "class": cls_name,
                        "confidence": round(float(conf), 4),
                        "box": [int(xmin), int(ymin), int(xmax), int(ymax)]
                    }
                )
            )

        return PredictionResponse(
            detections=detections,
            count=len(detections)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
