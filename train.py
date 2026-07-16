from ultralytics import YOLO
import torch

DATA_YAML = "dataset/data.yaml"

MODEL = "yolo11m.pt"

IMG_SIZE = 640
EPOCHS = 30
BATCH_SIZE = 32

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

model = YOLO(MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    workers=2,
    device=DEVICE,
    project="runs",
    name="wheel_detection",
    save=True,
    verbose=True
)

print("Training completed!")