from ultralytics import YOLO

DATA_YAML = "dataset/data.yaml"

MODEL = "yolo11n.pt"

IMG_SIZE = 640
EPOCHS = 20
BATCH_SIZE = 16

model = YOLO(MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    workers=2,
    device="cpu",
    project="runs",
    name="wheel_detection",
    save=True,
    verbose=True
)
