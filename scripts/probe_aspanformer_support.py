"""纯 CPU 的采样几何示例；不读取球数据、不运行 ASpanFormer 网络。"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def sample_points(centers, stds):
    """作者最细层的归一化前地址：2×2 query 均值，共用 8×8 样点。"""
    axis = np.arange(8, dtype=float) - 3.5
    offsets = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    scale = np.maximum(2 * stds * 5 / 8, 1).mean(axis=0)
    return centers.mean(axis=0) + offsets * scale


def support(samples, target):
    # ponytail: 单位 tent 仅表示理想细支撑；真实网络需另测其特征与最终匹配。
    response = np.prod(np.maximum(1 - np.abs(samples - target), 0), axis=1)
    return {
        "hull_contains_target": bool(
            np.all(samples.min(axis=0) <= target)
            and np.all(target <= samples.max(axis=0))
        ),
        "nearest_address_distance": float(np.linalg.norm(samples - target, axis=1).min()),
        "max_tent_response": float(response.max()),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/literature/aspanformer-span-geometry"
    output.mkdir(parents=True, exist_ok=True)
    centers = np.array([[24, 24], [25, 24], [24, 25], [56, 25]], dtype=float)
    stds = np.full_like(centers, 0.4)
    grouped = sample_points(centers, stds)
    own = sample_points(centers[-1:], stds[-1:])
    target = np.array([32, 32], dtype=float)
    narrow = sample_points(target[None], np.full((1, 2), 0.4))
    wide = sample_points(target[None], np.full((1, 2), 3.2))
    results = {
        "scope": "归一化前局部采样地址及理想连续 tent；不是实际特征或模型指标",
        "group_centers": centers.tolist(),
        "group_mean": centers.mean(axis=0).tolist(),
        "grouped_ball": support(grouped, centers[-1]),
        "ball_own_center": support(own, centers[-1]),
        "narrow_span": support(narrow, target),
        "wide_span": support(wide, target),
    }
    assert not results["grouped_ball"]["hull_contains_target"]
    assert results["ball_own_center"]["max_tent_response"] == 0.25
    assert results["narrow_span"]["hull_contains_target"]
    assert results["wide_span"]["hull_contains_target"]
    assert results["narrow_span"]["max_tent_response"] == 0.25
    assert results["wide_span"]["max_tent_response"] == 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.scatter(grouped[:, 0], grouped[:, 1], s=13, label="Shared 64 samples")
    ax.scatter(centers[:3, 0], centers[:3, 1], marker="^", color="gray",
               label="Other query addresses")
    ax.scatter(*centers[-1], marker="*", s=120, color="crimson", label="Ball query address")
    ax.scatter(*centers.mean(axis=0), marker="x", color="black", label="Shared mean")
    ax.set(title="Confident queries, different addresses", xlim=(20, 60), ylim=(16, 34))
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.scatter(wide[:, 0], wide[:, 1], s=15, color="darkorange", label="Spacing 4; 64 samples")
    ax.scatter(narrow[:, 0], narrow[:, 1], s=15, label="Spacing 1; 64 samples")
    ax.add_patch(Rectangle(target - 1, 2, 2, color="crimson", alpha=0.15))
    ax.scatter(*target, marker="*", s=90, color="crimson", label="Unit-support center")
    ax.set(title="Larger hull, no sample in tiny support", xlim=(15, 49), ylim=(15, 49))
    ax.legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.set(xlabel="Key-map x", ylabel="Key-map y", aspect="equal")
        ax.grid(alpha=0.15)
    figure = root / "doc/assets/aspanformer-span-geometry.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
