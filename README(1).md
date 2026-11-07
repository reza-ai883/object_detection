# YOLO11n vs YOLO11s: Object Detection on COCO128

Fine-tunes two YOLO11 models (nano and small) on the COCO128 dataset, evaluates them, and compares their real-time speed and detections on a live webcam feed.

## Project structure

```
.
├── train.py           # train + evaluate + plot curves (YOLO11n and YOLO11s)
├── webcam_demo.py     # real-time detection with one trained model
├── compare.py         # side-by-side live comparison of n and s
├── results/           # plots and CSV summaries
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

The COCO128 dataset and the pretrained `yolo11n.pt` / `yolo11s.pt` weights are downloaded automatically by Ultralytics on first run.

## Usage

### 1. Train and evaluate

```bash
python train.py                       # both models, 50 epochs
python train.py --models n --epochs 5 # quick test
```

Outputs:
- `runs/detect/yolo-n/` and `runs/detect/yolo-s/` (weights in `weights/best.pt`)
- `results/loss_curves_{n,s}.png`
- `results/performance_{n,s}_curves.png`
- `results/evaluation_summary_{n,s}.csv`
- `results/evaluation_comparison.csv`

Training is much faster with a GPU. Google Colab works well for this step:
download `best.pt` afterwards with `google.colab.files.download(...)`.

### 2. Live webcam detection

```bash
python webcam_demo.py --weights runs/detect/yolo-n/weights/best.pt
```

Press `Q` to quit. Use `--camera 1` if the default camera does not open.

### 3. Compare n vs s on the same frames

```bash
python compare.py
```

Both models run on every frame (alternating order to reduce bias). On quit, it prints average inference time, FPS, object count and confidence, and saves:
- `results/realtime_comparison.csv` (per frame)
- `results/comparison_summary.csv` (averages)

> **Note:** the webcam scripts need a local machine with a camera and display. They do not work on Colab or headless servers.

## Results

Add your plots and tables here after running, for example:

| Model   | mAP50 | mAP50-95 | Avg inference (ms) | Avg FPS |
|---------|-------|----------|--------------------|---------|
| YOLO11n |       |          |                    |         |
| YOLO11s |       |          |                    |         |

![Loss curves](results/loss_curves_n.png)

## Limitations

- COCO128 has only 128 images, and the same images are used for training and validation, so the metrics are optimistic. It is meant for quick experiments, not for real benchmarking.
- FPS depends heavily on hardware (CPU vs GPU), so compare models only within the same run.
