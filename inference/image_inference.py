"""SafeCityAI - Image Inference Module.

Performs YOLOv5 object detection on input images for traffic violation detection.
Classes: 0: Helmet, 1: LicensePlate, 2: NoHelmet.
"""

import argparse
import pathlib
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Fix for loading Linux/Colab-trained PyTorch checkpoints on Windows (Python 3.13+)
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath
    if hasattr(pathlib, "_local"):
        pathlib._local.PosixPath = pathlib.WindowsPath

# Define project root path relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "model" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Class ID mapping matching trained model and Roboflow dataset
CLASS_NAMES = {0: "Helmet", 1: "LicensePlate", 2: "NoHelmet"}
CLASS_COLORS = {
    0: (0, 255, 0),    # Helmet: Green
    1: (255, 165, 0),  # LicensePlate: Orange
    2: (0, 0, 255)     # NoHelmet: Red (Violation)
}


def load_model(weights_path: Path):
    """Load trained YOLOv5 PyTorch model with offline local repository support.

    Args:
        weights_path: Path to best.pt weights file.

    Returns:
        Loaded PyTorch YOLOv5 model object.

    Raises:
        FileNotFoundError: If weights file does not exist.
    """
    weights_path = Path(weights_path)
    print(f"[MODEL_LOAD] Checking weights file at: {weights_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            "Please train the model using Google Colab and place 'best.pt' in the model/ directory."
        )

    yolov5_dir = PROJECT_ROOT / "yolov5"
    model = None

    if yolov5_dir.is_dir() and (yolov5_dir / "hubconf.py").is_file():
        try:
            print(f"[MODEL_LOAD] Loading YOLOv5 model from local repository: {yolov5_dir}")
            model = torch.hub.load(
                str(yolov5_dir),
                'custom',
                path=str(weights_path),
                source='local',
                trust_repo=True
            )
            print("[MODEL_LOAD] Local offline model load successful!")
        except Exception as e:
            print(f"[MODEL_LOAD] Local model load failed ({e}), falling back to hub...")

    if model is None:
        print(f"[MODEL_LOAD] Loading YOLOv5 model via torch.hub...")
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=str(weights_path),
            trust_repo=True,
            force_reload=False
        )

    model.eval()
    return model


def run_image_inference(
    image_path: str,
    weights_path: str = str(DEFAULT_WEIGHTS_PATH),
    output_path: str = None,
    conf_threshold: float = 0.25
) -> dict:
    """Run object detection on a single image.

    Args:
        image_path: Path to input image file.
        weights_path: Path to model weights file.
        output_path: Destination path for annotated output image.
        conf_threshold: Minimum confidence threshold for detections.

    Returns:
        Dictionary containing detections list and output saved file path.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image file not found: {image_path}")

    # Load model gracefully
    model = load_model(Path(weights_path))
    model.conf = conf_threshold

    # Read image using OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image file: {image_path}")

    # Perform YOLOv5 inference
    results = model(img)

    # Parse predictions
    # results.xyxy[0] tensor format: [xmin, ymin, xmax, ymax, confidence, class_id]
    predictions = results.xyxy[0].cpu().numpy()

    detections = []
    annotated_img = img.copy()

    for pred in predictions:
        xmin, ymin, xmax, ymax, conf, cls_id = pred
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cls_id = int(cls_id)
        conf = float(conf)

        cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        # Record detection dictionary
        detections.append({
            "class": cls_name,
            "class_id": cls_id,
            "confidence": round(conf, 4),
            "box": [xmin, ymin, xmax, ymax]
        })

        # Draw bounding box
        cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), color, 2)

        # Draw label header
        label = f"{cls_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated_img,
            (xmin, max(ymin - label_h - 10, 0)),
            (xmin + label_w + 5, max(ymin, label_h + 10)),
            color,
            -1
        )
        cv2.putText(
            annotated_img,
            label,
            (xmin + 2, max(ymin - 5, label_h + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255) if color != (0, 255, 0) else (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    # Determine save destination
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"annotated_{image_path.name}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save output image
    cv2.imwrite(str(output_path), annotated_img)
    print(f"Saved annotated image to: {output_path}")

    return {
        "detections": detections,
        "count": len(detections),
        "output_path": str(output_path)
    }


def main():
    parser = argparse.ArgumentParser(description="SafeCityAI Image Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS_PATH), help="Path to best.pt model weights")
    parser.add_argument("--output", type=str, default=None, help="Output image file path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    try:
        res = run_image_inference(
            image_path=args.image,
            weights_path=args.weights,
            output_path=args.output,
            conf_threshold=args.conf
        )
        print(f"Inference Complete. Found {res['count']} detections.")
        for d in res["detections"]:
            print(f" - {d['class']}: {d['confidence']:.2f} at box {d['box']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
