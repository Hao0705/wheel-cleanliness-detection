Wheel Cleanliness Detection using YOLO11n
Overview

This project focuses on detecting and classifying the cleanliness of construction truck wheels using the YOLO11n object detection model. The model is fine-tuned from the pretrained yolo11n.pt checkpoint provided by Ultralytics.

The objective is to automatically identify whether each detected wheel is:

Clean
Dirty

This system can be integrated into vehicle monitoring systems at construction site exits to support automatic inspection and reduce environmental pollution caused by dirty truck wheels.

Model
Base model: yolo11n.pt
Framework: Ultralytics YOLO11
Task: Object Detection
Fine-tuning: Transfer Learning
Classes:
clean
dirty
Training Pipeline
Collect images of construction trucks.
Annotate wheel bounding boxes with two classes:
Clean
Dirty
Split the dataset into:
Train
Validation
Test
Fine-tune the pretrained YOLO11n model.
Evaluate the trained model on the test dataset.
Evaluation Results

The model was evaluated on 226 test images containing 796 wheel instances.

Metric	Value
Test Images	226
Total Instances	796
Precision (P)	0.994
Recall (R)	0.993
mAP@50	0.995
mAP@50-95	0.867
Per-Class Performance
Class	Images	Instances	Precision	Recall	mAP@50	mAP@50-95
Clean	116	356	0.997	0.989	0.995	0.852
Dirty	140	440	0.991	0.998	0.995	0.881
Inference Speed
Stage	Time / Image
Preprocess	1.3 ms
Inference	65.4 ms
Loss	0.0 ms
Postprocess	0.6 ms

Average inference time is approximately 67.3 ms per image, making the model suitable for near real-time deployment.

Results

The fine-tuned YOLO11n model achieved excellent detection performance:

Precision: 99.4%
Recall: 99.3%
mAP@50: 99.5%
mAP@50-95: 86.7%

The results demonstrate that transfer learning from YOLO11n provides highly accurate detection and classification of clean and dirty construction truck wheels while maintaining fast inference speed.
