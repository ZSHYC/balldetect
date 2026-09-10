"""保存固定规则选择的 GT/预测对照；图像产物不得用于模型输入。"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    with (args.run / "val_predictions.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["visibility_raw"] == "1"]
    for r in rows:
        for key in ("x_raw", "y_raw", "pred_x", "pred_y"):
            r[key] = float(r[key])
        r["error"] = np.hypot(r["x_raw"] - r["pred_x"], r["y_raw"] - r["pred_y"])
    rows.sort(key=lambda r: r["error"])
    # 同时展示正确与错误，挑选规则固定；不把此图当成总体统计。
    selected = rows[:2] + rows[-4:]
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3 * len(selected)), layout="constrained")
    for i, row in enumerate(selected):
        path = ROOT / "data/tracknet_tennis/Dataset" / row["game"] / row["clip"] / (row["original_frame_id"] + ".jpg")
        with Image.open(path) as im:
            image = im.convert("RGB")
        axes[i, 0].imshow(image)
        axes[i, 0].scatter(row["x_raw"], row["y_raw"], s=80, facecolors="none", edgecolors="lime", label="GT")
        axes[i, 0].scatter(row["pred_x"], row["pred_y"], s=80, marker="x", color="red", label="prediction")
        axes[i, 0].set_title(f"{row['clip']}/{row['original_frame_id']} error={row['error']:.1f}px")
        for j, (x, y, name) in enumerate(((row["x_raw"], row["y_raw"], "GT crop"),
                                         (row["pred_x"], row["pred_y"], "Predicted crop")), start=1):
            left, top = max(0, min(image.width - 64, round(x) - 32)), max(0, min(image.height - 64, round(y) - 32))
            axes[i, j].imshow(image.crop((left, top, left + 64, top + 64)), interpolation="nearest")
            axes[i, j].scatter(x - left, y - top, s=100, facecolors="none", edgecolors="lime" if j == 1 else "red")
            axes[i, j].set_title(name + " (64 original pixels)")
        for ax in axes[i]:
            ax.axis("off")
    fig.savefig(args.run / "error_examples.png", dpi=130)
    print(args.run / "error_examples.png")


if __name__ == "__main__":
    main()
