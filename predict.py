from ultralytics import YOLO

model = YOLO("./runs/wheel_detection/weights/best.pt")

model.predict(
    source="dataset/test/images",
    save_txt=True,
    save_conf=True,
    project="predict_result",
    name="labels_pred"
)