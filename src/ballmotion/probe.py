"""冻结空间特征的轻量读出及可复用的逐帧评价。"""
import math

import numpy as np
import torch
from torch import nn


class SpatialProbe(nn.Module):
    def __init__(self, channels, upscale=1, hidden_channels=0, num_frames=1, appearance_channels=None):
        super().__init__()
        self.upscale = upscale
        self.appearance_channels = channels if appearance_channels is None else appearance_channels
        # 拼接时一个完整帧占一组，避免历史帧尺度改变当前帧的归一化。
        self.norm = nn.GroupNorm(num_frames, self.appearance_channels, affine=False)
        if hidden_channels:
            self.location = nn.Sequential(nn.Conv2d(channels, hidden_channels, 1), nn.GELU(),
                                          nn.Conv2d(hidden_channels, upscale ** 2, 3, padding=1))
        else:
            self.location = nn.Conv2d(channels, upscale ** 2, 1)
        self.absence = nn.Linear(self.appearance_channels, 1)

    def forward(self, features):
        appearance = self.norm(features[:, :self.appearance_channels])
        features = (torch.cat((appearance, features[:, self.appearance_channels:]), dim=1)
                    if features.shape[1] > self.appearance_channels else appearance)
        spatial = torch.nn.functional.pixel_shuffle(self.location(features), self.upscale).flatten(1)
        absent = self.absence(appearance.mean((-2, -1))) + math.log(spatial.shape[1])
        return torch.cat((spatial, absent), dim=1)


def frame_batch(array, windows, mode, extra=None):
    if mode == "current":
        batch = array[windows[:, -1]]
    elif mode == "repeat":
        batch = np.tile(array[windows[:, -1]], (1, windows.shape[1], 1, 1))
    elif mode == "stack":
        b, t = windows.shape
        batch = array[windows].reshape(b, t * array.shape[1], *array.shape[-2:])
    else:
        raise ValueError(f"Unknown temporal input: {mode}")
    return np.concatenate((batch, extra), axis=1) if extra is not None else batch


def _location_metrics(errors):
    if not len(errors):
        return {"n": 0, "median_px": None, "mean_px": None,
                "pck8": None, "pck16": None, "pck32": None}
    return {"n": len(errors), "median_px": float(np.median(errors)),
            "mean_px": float(np.mean(errors)),
            **{f"pck{r}": float(np.mean(errors <= r)) for r in (8, 16, 32)}}


def _counts(tp, fp, fn):
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.}


def evaluate_predictions(rows, xy, presence_probability):
    xy = np.asarray(xy)
    visibility = np.array([r["visibility_raw"] for r in rows])
    present = visibility != 0
    target = np.array([[r["x_raw"], r["y_raw"]] for r in rows], dtype=float)
    if xy.shape != target.shape or len(presence_probability) != len(rows):
        raise ValueError("Prediction/target frame counts differ")
    if not np.isfinite(xy).all() or not np.isfinite(presence_probability).all():
        raise ValueError("Non-finite prediction")
    error = np.linalg.norm(xy - target, axis=1)
    detected = np.asarray(presence_probability) >= .5
    result = {"n_frames": len(rows), "location": _location_metrics(error[present]),
              "by_visibility": {str(v): _location_metrics(error[visibility == v]) for v in (1, 2, 3)},
              "presence": _counts(np.sum(present & detected), np.sum(~present & detected),
                                   np.sum(present & ~detected))}
    for radius in (8, 16, 32):
        correct = present & detected & (error <= radius)
        result[f"detection{radius}"] = _counts(np.sum(correct), np.sum(detected & ~correct),
                                               np.sum(present & ~correct))
    clip_ids = np.array([r["game"] + "/" + r["clip"] for r in rows])
    result["by_clip"] = {clip: _location_metrics(error[(clip_ids == clip) & present])
                         for clip in sorted(set(clip_ids))}
    return result


def paired_location_changes(rows, baseline_xy, challenger_xy):
    """共同目标上的条件定位净增益，不只统计被救回的样本。"""
    target = np.array([[r["x_raw"], r["y_raw"]] for r in rows], dtype=float)
    baseline_xy, challenger_xy = np.asarray(baseline_xy), np.asarray(challenger_xy)
    if baseline_xy.shape != target.shape or challenger_xy.shape != target.shape:
        raise ValueError("Paired predictions must cover the same targets")
    old_error = np.linalg.norm(baseline_xy - target, axis=1)
    new_error = np.linalg.norm(challenger_xy - target, axis=1)
    visibility = np.array([r["visibility_raw"] for r in rows])
    clips = np.array([r["game"] + "/" + r["clip"] for r in rows])
    groups = {"all": visibility != 0,
              **{f"visibility{v}": visibility == v for v in (1, 2, 3)},
              **{clip: (clips == clip) & (visibility != 0) for clip in sorted(set(clips))}}
    result = {}
    for name, mask in groups.items():
        result[name] = {}
        for radius in (8, 16, 32):
            old, new = old_error[mask] <= radius, new_error[mask] <= radius
            rescued, broken = int(np.sum(~old & new)), int(np.sum(old & ~new))
            result[name][str(radius)] = dict(n=int(mask.sum()), rescued=rescued, broken=broken,
                both_correct=int(np.sum(old & new)), both_wrong=int(np.sum(~old & ~new)),
                net_pck_change=(rescued - broken) / int(mask.sum()) if mask.any() else None)
    return result
