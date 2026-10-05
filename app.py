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
YOLOV5_DIR = PROJECT_ROOT / "yolov5"

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
    """Load PyTorch YOLOv5 model weights with resource caching and offline local fallback."""
    print(f"[MODEL_LOAD] [1/4] Checking weights file at: {WEIGHTS_PATH}")
    if not WEIGHTS_PATH.is_file():
        print(f"[MODEL_LOAD] [ERROR] Weights file missing at {WEIGHTS_PATH}")
        return None

    print(f"[MODEL_LOAD] [2/4] Checking local YOLOv5 directory at: {YOLOV5_DIR}")
    model = None

    print(f"[MODEL_LOAD] [3/4] Loading YOLOv5 model into PyTorch...")
    # Attempt offline load from bundled yolov5 directory first
    if YOLOV5_DIR.is_dir() and (YOLOV5_DIR / "hubconf.py").is_file():
        try:
            print(f"[MODEL_LOAD] Attempting offline load using local repository: {YOLOV5_DIR}...")
            model = torch.hub.load(
                str(YOLOV5_DIR),
                'custom',
                path=str(WEIGHTS_PATH),
                source='local',
                trust_repo=True
            )
            print("[MODEL_LOAD] Offline model loading from local repo SUCCESSFUL!")
        except Exception as e:
            print(f"[MODEL_LOAD] Local offline load failed ({e}), falling back to hub...")

    # Fallback to online torch.hub load if local fails
    if model is None:
        try:
            print("[MODEL_LOAD] Attempting torch.hub remote load (ultralytics/yolov5)...")
            model = torch.hub.load(
                'ultralytics/yolov5',
                'custom',
                path=str(WEIGHTS_PATH),
                trust_repo=True,
                force_reload=False
            )
            print("[MODEL_LOAD] Remote hub load SUCCESSFUL!")
        except Exception as e:
            print(f"[MODEL_LOAD] [ERROR] All model loading attempts failed: {e}")
            raise e

    if model is not None:
        model.eval()
        print(f"[MODEL_LOAD] [4/4] Model ready for inference. Type: {type(model)}")

    return model


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
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        # If file changed or not loaded yet in session state
        if st.session_state.get("uploaded_file_id") != file_id:
            try:
                image_bytes = uploaded_file.getvalue()
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.session_state["uploaded_file_id"] = file_id
                st.session_state["pil_image"] = pil_image
                st.session_state["annotated_image"] = None
                st.session_state["last_detections"] = None
            except Exception as e:
                st.error(f"Error opening uploaded image: {e}")
                return
    else:
        # File uploader cleared by user
        st.session_state.pop("uploaded_file_id", None)
        st.session_state.pop("pil_image", None)
        st.session_state.pop("annotated_image", None)
        st.session_state.pop("last_detections", None)

    # Render image & controls if PIL image exists in session state
    if "pil_image" in st.session_state and st.session_state["pil_image"] is not None:
        pil_image = st.session_state["pil_image"]
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(pil_image, use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Controls & Output")
            run_btn = st.button("🚀 Run Detection", type="primary", use_container_width=True)

            if run_btn:
                try:
                    with st.spinner("Loading model & running YOLOv5 traffic detection..."):
                        model = load_yolo_model()
                        if model is None:
                            st.error("Model weights `model/best.pt` missing or failed to load.")
                        else:
                            model.conf = conf_threshold

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
                            st.session_state["annotated_image"] = annotated_rgb
                            st.session_state["last_detections"] = detections
                except Exception as e:
                    st.error(f"Error running inference: {e}")

            # Display annotated output image if available
            if st.session_state.get("annotated_image") is not None:
                st.markdown("### 🔍 Annotated Detection Result")
                st.image(st.session_state["annotated_image"], use_container_width=True)
            elif not run_btn:
                st.info("Click **🚀 Run Detection** to perform inference on the uploaded image.")

        # Show Detection Summary Table if detection results exist in session state
        if st.session_state.get("last_detections") is not None:
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


