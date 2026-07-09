from ultralytics import YOLO
from pathlib import Path
import cv2

# =====================================================
# Config
# =====================================================

MODEL_PATH = "./runs/detect/runs/wheel_detection/weights/best.pt"

IMAGE_DIR = Path("./demo/images")
LABEL_DIR = Path("./demo/labels")
PREDICT_DIR = Path("./demo/predict")

CONF = 0.25

# =====================================================
# Create output folders
# =====================================================

LABEL_DIR.mkdir(exist_ok=True)
PREDICT_DIR.mkdir(exist_ok=True)

# =====================================================
# Load model
# =====================================================

model = YOLO(MODEL_PATH)

# =====================================================
# Supported image extensions
# =====================================================

image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.tif",
    "*.tiff"
]

image_files = []

for ext in image_extensions:
    image_files.extend(IMAGE_DIR.glob(ext))

print(f"Found {len(image_files)} images.")

# =====================================================
# Predict
# =====================================================

for image_path in image_files:

    print(f"Processing {image_path.name}")

    results = model.predict(
        source=str(image_path),
        conf=CONF,
        save=False,
        verbose=False
    )

    result = results[0]

    # ============================================
    # Save prediction image
    # ============================================

    plotted = result.plot()

    cv2.imwrite(
        str(PREDICT_DIR / image_path.name),
        plotted
    )

    # ============================================
    # Save txt (YOLO format)
    # ============================================

    txt_path = LABEL_DIR / (image_path.stem + ".txt")

    h, w = result.orig_shape

    with open(txt_path, "w") as f:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x_center, y_center, width, height = box.xywhn[0].tolist()

            f.write(
                f"{cls} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f} "
                f"{conf:.6f}\n"
            )

print("\nDone!")
print(f"Prediction images : {PREDICT_DIR}")
print(f"Prediction labels : {LABEL_DIR}")
