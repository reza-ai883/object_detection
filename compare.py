"""Compare YOLO11n and YOLO11s live on the same webcam frames.

Usage:
    python compare.py
    python compare.py --weights-n runs/detect/yolo-n/weights/best.pt \
                      --weights-s runs/detect/yolo-s/weights/best.pt
Press Q to quit. Results are written to results/ when you quit.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-n", default="runs/detect/yolo-n/weights/best.pt")
    parser.add_argument("--weights-s", default="runs/detect/yolo-s/weights/best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = until Q")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    models = {
        "YOLO11n": YOLO(args.weights_n),
        "YOLO11s": YOLO(args.weights_s),
    }

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Cannot open webcam (try another --camera index).")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        sys.exit("Cannot read from webcam.")

    # Warm-up so the first frame's load time doesn't skew the results
    for model in models.values():
        model.predict(frame, imgsz=args.imgsz, verbose=False)

    stats = {n: {"times": [], "fps": [], "counts": [], "confidence": []} for n in models}
    rows = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            outputs = {}

            # Alternate the order each frame to reduce order bias
            order = list(models.items())
            if frame_number % 2 == 0:
                order.reverse()

            for name, model in order:
                start = time.perf_counter()
                result = model.predict(
                    source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False
                )[0]
                inference_ms = (time.perf_counter() - start) * 1000

                fps = 1000 / inference_ms if inference_ms > 0 else 0
                count = len(result.boxes)
                avg_conf = float(result.boxes.conf.mean().item()) if count else 0.0

                s = stats[name]
                s["times"].append(inference_ms)
                s["fps"].append(fps)
                s["counts"].append(count)
                s["confidence"].append(avg_conf)

                rows.append({
                    "Frame": frame_number,
                    "Model": name,
                    "Inference_ms": inference_ms,
                    "FPS": fps,
                    "Objects": count,
                    "Mean_Confidence": avg_conf,
                })
                outputs[name] = result.plot()

            h, w = frame.shape[:2]
            combined = np.hstack([cv2.resize(outputs[n], (w, h)) for n in models])
            cv2.imshow("YOLO11n (left) vs YOLO11s (right)", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.max_frames and frame_number >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not rows:
        sys.exit("No frames were processed.")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    print("\n========== FINAL COMPARISON ==========")
    summary_rows = []
    for name, s in stats.items():
        avg = {
            "Average_Inference_ms": np.mean(s["times"]),
            "Average_FPS": np.mean(s["fps"]),
            "Average_Objects": np.mean(s["counts"]),
            "Average_Confidence": np.mean(s["confidence"]),
        }
        summary_rows.append({"Model": name, **avg})
        print(f"\n{name}")
        for k, v in avg.items():
            print(f"{k}: {v:.4f}")

    with open(results_dir / "realtime_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(results_dir / "comparison_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nResults saved in {results_dir}/")


if __name__ == "__main__":
    main()
