import pandas as pd
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import csv






# 1. Load trained YOLO-n model
model_n= YOLO("runs/detect/yolo-n/weights/best.pt")

# 2. Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

prev_time = 0

# 3. Real-Time Detection
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Object Detection
    results = model_n.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show output
    cv2.imshow("YOLO-n Real-Time Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Release resources
cap.release()
cv2.destroyAllWindows()


# 1. Load trained YOLO-s model
model_s= YOLO("runs/detect/yolo-s/weights/best.pt")

# 2. Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

prev_time = 0

# 3. Real-Time Detection
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Object Detection
    results = model_s.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show output
    cv2.imshow("YOLO-s Real-Time Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Release resources
cap.release()
cv2.destroyAllWindows()
