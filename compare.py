import pandas as pd
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import csv

# =====================================
# 1. Load Both Models
# =====================================

model_n = YOLO("runs/detect/yolo-n/weights/best.pt")
model_s = YOLO("runs/detect/yolo-s/weights/best.pt")

# =====================================
# 2. Open Webcam
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# =====================================
# 3. Prepare Results
# =====================================

results_data = []

stats = {
    "YOLO11n": {"times": [], "fps": [], "counts": [], "confidence": []},
    "YOLO11s": {"times": [], "fps": [], "counts": [], "confidence": []}
}

conf_threshold = 0.25
img_size = 640

# =====================================
# 4. Warm-up Models
# =====================================

ret, frame = cap.read()

if not ret:
    print("Cannot read webcam")
    cap.release()
    exit()

model_n.predict(frame, imgsz=img_size, verbose=False)
model_s.predict(frame, imgsz=img_size, verbose=False)

frame_number = 0

# =====================================
# 5. Real-Time Comparison
# =====================================
while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Same frame for both models
        outputs = {}

        # Alternate execution order to reduce order bias
        models = [
            ("YOLO11n", model_n),
            ("YOLO11s", model_s)
        ]

        if frame_number % 2 == 0:
            models.reverse()

        for name, model in models:

            start = time.perf_counter()

            result = model.predict(
                source=frame,
                conf=conf_threshold,
                imgsz=img_size,
                verbose=False
            )[0]

            end = time.perf_counter()

            # Inference time
            inference_time = (end - start) * 1000

            # Inference FPS
            fps = 1000 / inference_time if inference_time > 0 else 0

            # Number of detected objects
            object_count = len(result.boxes)

            # Average confidence
            if object_count > 0:
                avg_conf = float(result.boxes.conf.mean().item())
            else:
                avg_conf = 0.0

            # Save statistics
            stats[name]["times"].append(inference_time)
            stats[name]["fps"].append(fps)
            stats[name]["counts"].append(object_count)
            stats[name]["confidence"].append(avg_conf)

            # Save frame results
            results_data.append({
                "Frame": frame_number,
                "Model": name,
                "Inference_ms": inference_time,
                "FPS": fps,
                "Objects": object_count,
                "Mean_Confidence": avg_conf
            })

            # Draw detections
            annotated = result.plot()

            outputs[name] = annotated

        # =====================================
        # 6. Display Both Models
        # =====================================

        frame_n = outputs["YOLO11n"]
        frame_s = outputs["YOLO11s"]

        # Resize to same dimensions
        h, w = frame.shape[:2]

        frame_n = cv2.resize(frame_n, (w, h))
        frame_s = cv2.resize(frame_s, (w, h))

        # Combine images horizontally
        combined = np.hstack((frame_n, frame_s))

        # Display
        cv2.imshow("YOLO11n vs YOLO11s", combined)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

# =====================================
# 7. Calculate Final Statistics
# =====================================

print("\n========== FINAL COMPARISON ==========")

for name in ["YOLO11n", "YOLO11s"]:

    avg_time = np.mean(stats[name]["times"])
    avg_fps = np.mean(stats[name]["fps"])
    avg_objects = np.mean(stats[name]["counts"])
    avg_conf = np.mean(stats[name]["confidence"])

    print(f"\n{name}")

    print("Average Inference Time:", round(avg_time, 2), "ms")
    print("Average FPS:", round(avg_fps, 2))
    print("Average Objects:", round(avg_objects, 2))
    print("Average Confidence:", round(avg_conf, 4))

# =====================================
# 8. Save Frame-by-Frame Results
# =====================================

with open("realtime_comparison.csv", "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Frame",
            "Model",
            "Inference_ms",
            "FPS",
            "Objects",
            "Mean_Confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(results_data)

# =====================================
# 9. Save Final Summary
# =====================================

with open("comparison_summary.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Average_Inference_ms",
        "Average_FPS",
        "Average_Objects",
        "Average_Confidence"
    ])

    for name in ["YOLO11n", "YOLO11s"]:

        writer.writerow([
            name,
            np.mean(stats[name]["times"]),
            np.mean(stats[name]["fps"]),
            np.mean(stats[name]["counts"]),
            np.mean(stats[name]["confidence"])
        ])

print("\nComparison completed successfully!")
print("Results saved in CSV files.")