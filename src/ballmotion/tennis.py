"""Tennis 发布索引与像素中心几何。此处不套用其他数据集的缺标规则。"""
import csv
import math
from pathlib import Path

import numpy as np


def label_state(visibility, x, y, width, height):
    if visibility is None:
        return "unknown"
    if visibility == 0:
        return "absent"
    if visibility not in (1, 2, 3):
        raise ValueError(f"Unknown Tennis visibility: {visibility}")
    if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
        return "invalid"
    return "located" if 0 <= x < width and 0 <= y < height else "invalid"


def read_frames(data_root, games, frame_step=1):
    if frame_step < 1:
        raise ValueError("frame_step must be positive")
    frames, excluded = [], []
    with (Path(data_root) / "frames.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["game"][4:]) not in games:
                continue
            if int(row["original_frame_id"]) % frame_step:
                continue
            for key in ("visibility_raw", "width", "height"):
                row[key] = int(row[key])
            for key in ("x_raw", "y_raw"):
                row[key] = float(row[key]) if row[key] else None
            row["label_state"] = label_state(row["visibility_raw"], row["x_raw"], row["y_raw"],
                                               row["width"], row["height"])
            if row["label_state"] in ("invalid", "unknown"):
                excluded.append(row)
            else:
                frames.append(row)
    frames.sort(key=lambda r: (int(r["game"][4:]), int(r["clip"][4:]), int(r["original_frame_id"])))
    return frames, excluded


def grid_targets(xy, present, grid_hw, original_hw=(720, 1280)):
    h, w = grid_hw
    oh, ow = original_hw
    xy = np.asarray(xy, dtype=float)
    xy = np.where(np.asarray(present, dtype=bool)[:, None], xy, 0.)
    gx = np.floor((xy[:, 0] + .5) * w / ow).astype(np.int64)
    gy = np.floor((xy[:, 1] + .5) * h / oh).astype(np.int64)
    return np.where(present, gy * w + gx, h * w)


def grid_to_original(indices, grid_hw, original_hw=(720, 1280)):
    h, w = grid_hw
    oh, ow = original_hw
    indices = np.asarray(indices)
    return np.stack(((indices % w + .5) * ow / w - .5,
                     (indices // w + .5) * oh / h - .5), axis=-1)
