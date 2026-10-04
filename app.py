"""SafeCityAI - Streamlit Live Web Application.

Interactive web frontend for YOLOv5 real-time traffic violation detection.
Detects: 0: Helmet, 1: LicensePlate, 2: NoHelmet.
"""

import io
import pathlib
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Fix for loading Linux/Colab-trained PyTorch checkpoints on Windows (Python 3.13+)
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath
    if hasattr(pathlib, "_local"):
        pathlib._local.PosixPath = pathlib.WindowsPath

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "model" / "best.pt"

# Class ID mapping matching trained model and Roboflow dataset
CLASS_NAMES = {0: "Helmet", 1: "LicensePlate", 2: "NoHelmet"}
CLASS_COLORS = {
    0: (0, 255, 0),    # Helmet: Green
    1: (255, 165, 0),  # LicensePlate: Orange
    2: (0, 0, 255)     # NoHelmet: Red (Violation)
}

# Streamlit Page Configuration
st.set_page_config(
    page_title="SafeCityAI – Traffic Violation Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_yolo_model():
    """Load PyTorch YOLOv5 model weights with resource caching."""
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
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None


def main():
    st.title("🚦 SafeCityAI – Traffic Violation Detection")
    st.markdown(
        "Automated computer vision system for traffic-rule enforcement using **YOLOv5 transfer learning**. "
        "Upload a traffic image to detect protective helmets, identify helmet-less violations, and locate vehicle license plates."
    )

    # Sidebar Configuration & Class Information
    st.sidebar.header("⚙️ Control Panel")

    if WEIGHTS_PATH.is_file():
        st.sidebar.success(f"Model Ready: `model/best.pt` ({WEIGHTS_PATH.stat().st_size / (1024*1024):.1f} MB)")
    else:
        st.sidebar.error("Model weights `model/best.pt` not found!")

    st.sidebar.markdown("### Supported Target Classes:")
    st.sidebar.markdown("- 🟢 **Helmet** (Compliant Rider)")
    st.sidebar.markdown("- 🟠 **LicensePlate** (Vehicle Identification)")
    st.sidebar.markdown("- 🔴 **NoHelmet** (Traffic Violation)")

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Minimum confidence score threshold for object detection."
    )

    st.sidebar.markdown("---")
    st.sidebar.info("SafeCityAI Capstone Project | YOLOv5 Transfer Learning")

    # Image Uploader Component
    uploaded_file = st.file_uploader(
        "Upload a traffic image for violation detection (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Load uploaded image bytes
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(pil_image, use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Controls & Output")
            run_btn = st.button("🚀 Run Detection", type="primary", use_container_width=True)

            if run_btn:
                model = load_yolo_model()

                if model is None:
                    st.error("Model weights `model/best.pt` missing or unavailable.")
                    return

                model.conf = conf_threshold

                with st.spinner("Running YOLOv5 traffic detection..."):
                    img_np = np.array(pil_image)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    results = model(img_bgr)
                    predictions = results.xyxy[0].cpu().numpy()

                    annotated_bgr = img_bgr.copy()
                    detections = []

                    for pred in predictions:
                        xmin, ymin, xmax, ymax, conf, cls_id = pred
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                        cls_id = int(cls_id)
                        conf = float(conf)

                        cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
                        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                        detections.append({
                            "Class": cls_name,
                            "Confidence": f"{conf:.2%}",
                            "Bounding Box [xmin, ymin, xmax, ymax]": f"[{xmin}, {ymin}, {xmax}, {ymax}]"
                        })

                        # Draw bounding box rectangle
                        cv2.rectangle(annotated_bgr, (xmin, ymin), (xmax, ymax), color, 2)

                        # Draw text label banner
                        label = f"{cls_name} {conf:.2f}"
                        (label_w, label_h), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            annotated_bgr,
                            (xmin, max(ymin - label_h - 10, 0)),
                            (xmin + label_w + 5, max(ymin, label_h + 10)),
                            color,
                            -1
                        )
                        cv2.putText(
                            annotated_bgr,
                            label,
                            (xmin + 2, max(ymin - 5, label_h + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255) if color != (0, 255, 0) else (0, 0, 0),
                            1,
                            cv2.LINE_AA
                        )

                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)

                    st.session_state["last_detections"] = detections
            else:
                st.info("Click **🚀 Run Detection** to perform inference on the uploaded image.")

        # Show Detection Summary Table if detection was run
        if "last_detections" in st.session_state and run_btn:
            st.markdown("---")
            st.subheader("📊 Detection Summary Table")
            detections_list = st.session_state["last_detections"]

            if detections_list:
                df = pd.DataFrame(detections_list)
                st.dataframe(df, use_container_width=True)
                st.success(f"Detected {len(detections_list)} target object(s) above threshold {conf_threshold:.2f}.")
            else:
                st.info("No detections found above the selected confidence threshold.")

    else:
        st.info("👆 Upload a traffic image file (JPG, JPEG, PNG) to begin violation detection.")


if __name__ == "__main__":
    main()
