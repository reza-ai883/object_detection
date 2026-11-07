"""Train and evaluate YOLO11n and YOLO11s on COCO128.

Usage:
    python train.py                      # both models, 50 epochs
    python train.py --models n --epochs 5
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

WEIGHTS = {"n": "yolo11n.pt", "s": "yolo11s.pt"}


def plot_curves(df, title, out_path, columns):
    plt.figure(figsize=(12, 6))
    for col, label in columns:
        plt.plot(df["epoch"], df[col], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_and_eval(key, data, epochs, imgsz, results_dir):
    name = f"yolo-{key}"
    print(f"\n===== Training {name} =====")

    model = YOLO(WEIGHTS[key])
    model.train(data=data, epochs=epochs, imgsz=imgsz, name=name, exist_ok=True)

    # Use the real output folder instead of a hard-coded path
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    print(f"\n===== Evaluating {name} =====")
    metrics = YOLO(best).val(data=data, plots=True, name=f"{name}-val", exist_ok=True)

    df = pd.read_csv(save_dir / "results.csv")
    df.columns = df.columns.str.strip()
    last = df.iloc[-1]

    plot_curves(
        df,
        f"{name} - Training and Validation Loss",
        results_dir / f"loss_curves_{key}.png",
        [
            ("train/box_loss", "Train Box Loss"),
            ("val/box_loss", "Validation Box Loss"),
            ("train/cls_loss", "Train Classification Loss"),
            ("val/cls_loss", "Validation Classification Loss"),
        ],
    )
    plot_curves(
        df,
        f"{name} - Performance During Training",
        results_dir / f"performance_{key}_curves.png",
        [
            ("metrics/precision(B)", "Precision"),
            ("metrics/recall(B)", "Recall"),
            ("metrics/mAP50(B)", "mAP50"),
            ("metrics/mAP50-95(B)", "mAP50-95"),
        ],
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Precision", "Recall", "mAP50", "mAP50-95",
                "Train Box Loss", "Train Cls Loss", "Val Box Loss", "Val Cls Loss",
            ],
            "Value": [
                metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map,
                last["train/box_loss"], last["train/cls_loss"],
                last["val/box_loss"], last["val/cls_loss"],
            ],
        }
    )
    summary.to_csv(results_dir / f"evaluation_summary_{key}.csv", index=False)
    print(summary)
    print(f"Best weights: {best}")
    return summary.set_index("Metric")["Value"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--models", nargs="+", choices=list(WEIGHTS), default=["n", "s"])
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    all_summaries = {}
    for key in args.models:
        all_summaries[f"YOLO11{key}"] = train_and_eval(
            key, args.data, args.epochs, args.imgsz, results_dir
        )

    if len(all_summaries) > 1:
        table = pd.DataFrame(all_summaries)
        table.to_csv(results_dir / "evaluation_comparison.csv")
        print("\n===== Final comparison =====")
        print(table)

    print("\nAll evaluation tasks completed!")


if __name__ == "__main__":
    main()
