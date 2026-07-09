# Wheel Cleanliness Detection using YOLO11n

## Overview

This project focuses on detecting and classifying the cleanliness of construction truck wheels using the **YOLO11n** object detection model. The model is fine-tuned from the pretrained **yolo11n.pt** checkpoint provided by Ultralytics.

The objective is to automatically identify whether each detected wheel is:

- Clean
- Dirty

This system can be integrated into vehicle monitoring systems at construction site exits to support automatic inspection and reduce environmental pollution caused by dirty truck wheels.

---

## Model

- **Base model:** `yolo11n.pt`
- **Framework:** Ultralytics YOLO11
- **Task:** Object Detection
- **Fine-tuning:** Transfer Learning

### Classes

- Clean
- Dirty

---

## Training Pipeline

1. Collect images of construction trucks.
2. Annotate wheel bounding boxes with two classes:
   - Clean
   - Dirty
3. Split the dataset into:
   - Train
   - Validation
   - Test
4. Fine-tune the pretrained YOLO11n model.
5. Evaluate the trained model on the test dataset.

---

## Evaluation Results

The model was evaluated on **228 test images** containing **797 wheel instances**.

| Metric | Value |
|--------|------:|
| Test Images | 228 |
| Total Instances | 797 |
| Precision (P) | 0.995 |
| Recall (R) | 0.987 |
| mAP@50 | 0.994 |
| mAP@50-95 | 0.856 |

---

## Per-Class Performance

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|------|------:|----------:|----------:|--------:|--------:|-----------:|
| Clean | 120 | 344 | 0.997 | 0.983 | 0.992 | 0.856 |
| Dirty | 153 | 453 | 0.992 | 0.991 | 0.995 | 0.856 |

---

## Inference Speed

| Stage | Time / Image |
|--------|-------------:|
| Preprocess | 1.3 ms |
| Inference | 53.9 ms |
| Loss | 0.0 ms |
| Postprocess | 0.6 ms |

Average inference time is approximately **55.8 ms per image**, making the model suitable for near real-time deployment.

---

## Results

The fine-tuned YOLO11n model achieved excellent detection performance:

- **Precision:** 99.5%
- **Recall:** 98.7%
- **mAP@50:** 99.4%
- **mAP@50-95:** 85.6%

The results demonstrate that transfer learning from **YOLO11n** enables highly accurate detection and classification of clean and dirty construction truck wheels while maintaining fast inference speed, making the model suitable for practical deployment in construction site vehicle inspection systems.

The results demonstrate that transfer learning from **YOLO11n** provides highly accurate detection and classification of clean and dirty construction truck wheels while maintaining fast inference speed.
