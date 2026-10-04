# SafeCityAI Dataset Guide

This directory is configured for training the **SafeCityAI** traffic violation detection model using YOLOv5.

## Expected Directory Structure

When preparing your custom dataset for training, structure your dataset folder as follows:

```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

---

## YOLO Annotation Format

Each image in `images/train` and `images/val` must have a corresponding `.txt` label file in `labels/train` and `labels/val` with the **exact same base name**.

Each line in the `.txt` label file defines one bounding box in the following format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

### Parameters
- **`class_id`**: Integer index corresponding to the class name in `data.yaml`:
  - `0`: `Helmet`
  - `1`: `NoHelmet`
  - `2`: `LicensePlate`
- **`x_center`**: Bounding box center X coordinate, normalized relative to image width (`0.0` to `1.0`).
- **`y_center`**: Bounding box center Y coordinate, normalized relative to image height (`0.0` to `1.0`).
- **`width`**: Bounding box width, normalized relative to image width (`0.0` to `1.0`).
- **`height`**: Bounding box height, normalized relative to image height (`0.0` to `1.0`).

### Example Label File (`image1.txt`)
```text
0 0.456 0.234 0.120 0.180
2 0.460 0.780 0.250 0.100
1 0.780 0.310 0.140 0.210
```

---

## Data Split Recommendation

- **Training set (`train`)**: ~80% of total annotated images.
- **Validation set (`val`)**: ~20% of total annotated images.
- Ensure balanced representation of all three classes across both splits.
