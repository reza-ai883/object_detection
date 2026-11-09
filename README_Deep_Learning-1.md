# Real-Time Object Detection and Model Comparison Using YOLO11

## Overview

This project explores object detection using the Ultralytics YOLO11 architecture.

Two YOLO11 model variants are trained and evaluated on the COCO128 dataset. Their performance is examined through object-detection evaluation metrics, and the trained models are also used for real-time inference with a webcam.

## Dataset

The project uses COCO128, a compact object-detection dataset containing images with annotated object classes and bounding boxes.

## Technologies

- Python
- PyTorch
- Ultralytics YOLO11
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Main Features

- Training two YOLO11 model variants
- Model validation and evaluation
- Object detection on images
- Real-time object detection using a webcam
- Evaluation using precision, recall, mAP@50, and mAP@50-95
- Analysis of training and validation losses
- Comparison of model performance

## Training and Evaluation Workflow

1. Load the COCO128 dataset configuration.
2. Initialize and train the YOLO11 model variants.
3. Validate the trained models.
4. Collect object-detection evaluation metrics.
5. Review training and validation loss values.
6. Run inference on images and webcam video.
7. Compare the models based on their evaluation results and observed inference performance.

## Evaluation Metrics

The project uses standard object-detection metrics:

- **Precision:** The proportion of predicted detections that are correct.
- **Recall:** The proportion of ground-truth objects successfully detected.
- **mAP@50:** Mean average precision at an intersection-over-union (IoU) threshold of 0.50.
- **mAP@50-95:** Mean average precision averaged over IoU thresholds from 0.50 to 0.95.

Training and validation losses are also reviewed to understand model learning behavior.

## Conclusion

This project demonstrates a practical object-detection workflow using YOLO11, including model training, validation, metric-based evaluation, and real-time webcam inference. Comparing two model variants provides a basis for examining differences in detection performance.
