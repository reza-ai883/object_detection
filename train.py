import pandas as pd
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os



model_n = YOLO("yolo11n.pt")
model_n.train(data="coco128.yaml",epochs=50,imgsz=640,name="yolo-n",exist_ok=True)


metrics_n = model_n.val(data="coco128.yaml",plots=True)

print("\n========== Evaluation Metrics ==========")

print("Precision:", metrics_n.box.mp)
print("Recall:", metrics_n.box.mr)
print("mAP50:", metrics_n.box.map50)
print("mAP50-95:", metrics_n.box.map)



results_path = "runs/detect/yolo-n/results.csv"

df_n = pd.read_csv(results_path)

df_n.columns = df_n.columns.str.strip()

print("\n========== Available Columns ==========")
print(df_n.columns.tolist())



print("\n========== Final Epoch Losses ==========")

last = df_n.iloc[-1]

print("Train Box Loss:", last["train/box_loss"])
print("Train Cls Loss:", last["train/cls_loss"])
print("Val Box Loss:", last["val/box_loss"])
print("Val Cls Loss:", last["val/cls_loss"])



plt.figure(figsize=(12, 6))

plt.plot(df_n["epoch"], df_n["train/box_loss"],
         label="Train Box Loss")

plt.plot(df_n["epoch"], df_n["val/box_loss"],
         label="Validation Box Loss")

plt.plot(df_n["epoch"], df_n["train/cls_loss"],
         label="Train Classification Loss")

plt.plot(df_n["epoch"], df_n["val/cls_loss"],
         label="Validation Classification Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation yolo-n_Loss")

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("loss_curves_n.png")
plt.show()

# =====================================
# 6. Precision, Recall and mAP Plot
# =====================================

plt.figure(figsize=(12, 6))

plt.plot(df_n["epoch"], df_n["metrics/precision(B)"],
         label="Precision")

plt.plot(df_n["epoch"], df_n["metrics/recall(B)"],
         label="Recall")

plt.plot(df_n["epoch"], df_n["metrics/mAP50(B)"],
         label="mAP50")

plt.plot(df_n["epoch"], df_n["metrics/mAP50-95(B)"],
         label="mAP50-95")

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("yolo-n Performance During Training")

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("performance_n_curves.png")
plt.show()

# =====================================
# 7. Save Evaluation Summary
# =====================================

summary = pd.DataFrame({
    "Metric_n": [
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
        "Train Box Loss",
        "Train Cls Loss",
        "Val Box Loss",
        "Val Cls Loss"
    ],

    "Value": [
        metrics_n.box.mp,
        metrics_n.box.mr,
        metrics_n.box.map50,
        metrics_n.box.map,
        last["train/box_loss"],
        last["train/cls_loss"],
        last["val/box_loss"],
        last["val/cls_loss"]
    ]
})

summary.to_csv("evaluation_summary_n.csv", index=False)

print("\n========== Final Summary ==========")
print(summary)

print("\nAll evaluation tasks completed!")