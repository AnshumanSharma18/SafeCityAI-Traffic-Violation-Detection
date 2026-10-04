"""SafeCityAI - Video Inference Module.

Performs frame-by-frame YOLOv5 object detection on input videos for traffic violation enforcement.
Classes: 0: Helmet, 1: LicensePlate, 2: NoHelmet.
"""

import argparse
import pathlib
import sys
from pathlib import Path
import cv2
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
    """Load trained YOLOv5 PyTorch model.

    Args:
        weights_path: Path to best.pt weights file.

    Returns:
        Loaded PyTorch YOLOv5 model object.

    Raises:
        FileNotFoundError: If weights file does not exist.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            "Please train the model using Google Colab and place 'best.pt' in the model/ directory."
        )

    print(f"Loading YOLOv5 model from: {weights_path}")
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=str(weights_path),
        trust_repo=True,
        force_reload=False
    )
    return model


def run_video_inference(
    video_path: str,
    weights_path: str = str(DEFAULT_WEIGHTS_PATH),
    output_path: str = None,
    conf_threshold: float = 0.25
) -> dict:
    """Run object detection on an MP4/AVI video stream frame-by-frame.

    Args:
        video_path: Path to input video file.
        weights_path: Path to model weights file.
        output_path: Destination path for annotated MP4 output video.
        conf_threshold: Minimum confidence threshold.

    Returns:
        Dictionary containing video metrics and output file path.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    # Load YOLOv5 model
    model = load_model(Path(weights_path))
    model.conf = conf_threshold

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0  # Fallback FPS if unreadable

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output path
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"annotated_{video_path.stem}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Processing Video: {video_path.name}")
    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")

    frame_count = 0
    total_detections_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run YOLOv5 inference on frame
            results = model(frame)
            predictions = results.xyxy[0].cpu().numpy()

            total_detections_count += len(predictions)

            # Draw detections
            for pred in predictions:
                xmin, ymin, xmax, ymax, conf, cls_id = pred
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                cls_id = int(cls_id)
                conf = float(conf)

                cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw text label
                label = f"{cls_name} {conf:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame,
                    (xmin, max(ymin - label_h - 10, 0)),
                    (xmin + label_w + 5, max(ymin, label_h + 10)),
                    color,
                    -1
                )
                cv2.putText(
                    frame,
                    label,
                    (xmin + 2, max(ymin - 5, label_h + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255) if color != (0, 255, 0) else (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

            # Write processed frame to output video file
            writer.write(frame)

            if frame_count % 30 == 0 or frame_count == total_frames:
                print(f"Processed frame {frame_count}/{total_frames}...")

    finally:
        cap.release()
        writer.release()

    print(f"Finished processing. Output video saved to: {output_path}")

    return {
        "total_frames": frame_count,
        "total_detections": total_detections_count,
        "fps": fps,
        "resolution": [width, height],
        "output_path": str(output_path)
    }


def main():
    parser = argparse.ArgumentParser(description="SafeCityAI Video Inference Script")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS_PATH), help="Path to best.pt model weights")
    parser.add_argument("--output", type=str, default=None, help="Output video file path (.mp4)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    try:
        res = run_video_inference(
            video_path=args.video,
            weights_path=args.weights,
            output_path=args.output,
            conf_threshold=args.conf
        )
        print(f"Video Processing Complete! Frames: {res['total_frames']}, Total Detections: {res['total_detections']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
