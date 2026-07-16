from ultralytics import YOLO

# ==================================================
# CONFIG
# ==================================================

# Mô hình tốt nhất sau khi train
MODEL_PATH = "./runs/wheel_detection/weights/best.pt"

# File data.yaml
DATA_YAML = "dataset/data.yaml"

# ==================================================
# Load model
# ==================================================

model = YOLO(MODEL_PATH)

# ==================================================
# Đánh giá trên tập test
# ==================================================

metrics = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=16,
    project="ketqua",
    name="evaluation"
)

# ==================================================
# Hiển thị kết quả
# ==================================================

print("=" * 60)
print("Evaluation on Test Dataset")
print("=" * 60)

print(f"Precision (P) : {metrics.box.mp:.4f}")
print(f"Recall (R)    : {metrics.box.mr:.4f}")
print(f"mAP@0.5       : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")

print("=" * 60)
