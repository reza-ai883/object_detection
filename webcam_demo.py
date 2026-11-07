"""Real-time object detection from a webcam with one trained model.

Usage:
    python webcam_demo.py --weights runs/detect/yolo-n/weights/best.pt
Press Q to quit.
"""
import argparse
import sys
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to best.pt")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Cannot open webcam (try another --camera index).")

    window = "YOLO Real-Time Detection"
    prev_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model.predict(
                source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False
            )[0]
            annotated = result.plot()

            now = time.time()
            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now

            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.imshow(window, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
