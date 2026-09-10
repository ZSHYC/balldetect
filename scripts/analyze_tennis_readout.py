"""分析已选DINO验证读出的存在分数与条件空间集中度。"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
from ballmotion.backbone_probe import BackboneProbe
from ballmotion.probe import SpatialProbe
from ballmotion.tennis import grid_to_original
from compare_predictions import read_predictions
from dinov3.models.convnext import ConvNeXt


SCORE_NAMES = ("q", "maxc", "entropy_normalized", "m16", "q_times_m16")


def readout_scores(logits, grid_hw):
    height, width = grid_hw
    if logits.ndim != 2 or logits.shape[1] != height * width + 1:
        raise ValueError("logits 与位置格/absence类别不匹配")
    location_logits = logits[:, :-1]
    joint = logits.softmax(1)
    conditional = location_logits.softmax(1)
    predicted = location_logits.argmax(1)
    neighborhood = F.avg_pool2d(conditional.reshape(-1, 1, height, width), 3,
                                stride=1, padding=1, divisor_override=1).flatten(1)
    q = 1 - joint[:, -1]
    m16 = neighborhood.gather(1, predicted[:, None]).squeeze(1)
    return {
        "predicted_cell": predicted,
        "q": q,
        "maxc": conditional.max(1).values,
        "entropy_normalized": torch.special.entr(conditional).sum(1) / math.log(height * width),
        "m16": m16,
        "q_times_m16": q * m16,
    }


def auroc(scores, positive):
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return None
    differences = pos[:, None] - neg[None, :]
    return float(np.mean(differences > 0) + .5 * np.mean(differences == 0))


def score_summary(score_arrays, mask):
    mask = np.asarray(mask, dtype=bool)
    result = {"n": int(mask.sum())}
    for name in SCORE_NAMES:
        values = score_arrays[name][mask]
        result[name] = ({"p10": float(np.quantile(values, .1)),
                         "median": float(np.median(values)),
                         "p90": float(np.quantile(values, .9))} if len(values) else
                        {"p10": None, "median": None, "p90": None})
    return result


def build_model(config, checkpoint_path, device):
    model_config = config["model_config"]
    if config.get("model") != "dino" or model_config.get("backbone") != "dinov3_convnext_tiny":
        raise ValueError("只支持已完成的DINOv3 ConvNeXt-Tiny读出")
    backbone = ConvNeXt(depths=model_config["depths"], dims=model_config["dims"])
    prefix = torch.nn.Sequential(backbone.downsample_layers[0], backbone.stages[0],
                                 backbone.downsample_layers[1], backbone.stages[1])
    del backbone
    head_config = model_config["head"]
    head = SpatialProbe(head_config["input_channels"], upscale=head_config["upscale"],
                        hidden_channels=head_config["hidden_channels"],
                        num_frames=head_config["num_frames"],
                        appearance_channels=head_config["appearance_channels"])
    model = BackboneProbe(prefix, head, train_backbone=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), int(checkpoint["epoch"])


def row_identity(row):
    return (row["game"], row["clip"], str(row["original_frame_id"]),
            int(row["visibility_raw"]), row["x_raw"], row["y_raw"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = ROOT / args.run
    output_csv = run / "readout_concentration.csv"
    output_json = run / "readout_concentration.json"
    config = json.loads((run / "config.json").read_text())
    results = json.loads((run / "results.json").read_text())
    checkpoint_path = run / "best.pt"
    saved_rows, saved_xy, saved_q = read_predictions(run / "val_predictions.csv")

    rgb_cache = ROOT / config["rgb_cache"]
    metadata = json.loads((rgb_cache / "metadata.json").read_text())
    frames = metadata["frames"]
    if set(row["game"] for row in frames) - {f"game{i}" for i in range(1, 8)}:
        raise ValueError("RGB缓存包含保留的最终测试比赛")
    windows = np.asarray(metadata["windows"], dtype=np.int64)
    if windows.ndim != 2 or windows.shape[1] != 3:
        raise ValueError("分析需要原始三帧因果窗口")
    temporal_input = config.get("temporal_input", "history")
    if temporal_input not in ("history", "repeat_current"):
        raise ValueError(f"未知DINO输入模式: {temporal_input}")
    input_slots = (["t-2", "t-1", "t"] if temporal_input == "history" else ["t", "t", "t"])
    if config.get("input_slots", input_slots) != input_slots:
        raise ValueError("config中的temporal_input与input_slots不一致")
    model_windows = (windows if temporal_input == "history"
                     else np.repeat(windows[:, -1:], 3, axis=1))
    rows = [frames[int(window[-1])] for window in windows]
    val_indices = np.asarray([index for index, row in enumerate(rows) if row["game"] == "game7"])
    val_rows = [rows[index] for index in val_indices]
    if [row_identity(row) for row in val_rows] != [row_identity(row) for row in saved_rows]:
        raise ValueError("缓存与既有val_predictions的目标身份、顺序或标签不同")

    rgb = np.load(rgb_cache / "rgb.npy", mmap_mode="r")
    if rgb.shape != (len(frames), 3, 288, 512) or rgb.dtype != np.uint8:
        raise ValueError("RGB缓存必须为N×3×288×512 uint8")
    device = torch.device("cuda")
    model, checkpoint_epoch = build_model(config, checkpoint_path, device)
    if checkpoint_epoch != int(results["best_epoch"]):
        raise ValueError("best.pt epoch与results.json选中epoch不同")
    grid_hw = tuple(config["output_grid_hw"])
    if grid_hw != (72, 128):
        raise ValueError("该分析固定使用72×128 DINO位置格")

    started = time.perf_counter()
    score_chunks = {name: [] for name in SCORE_NAMES}
    predicted_cells = []
    batch_size = int(config["batch_size"])
    with torch.inference_mode():
        for start in range(0, len(val_indices), batch_size):
            ids = val_indices[start:start + batch_size]
            batch_windows = model_windows[ids]
            unique_frames, inverse = np.unique(batch_windows, return_inverse=True)
            features = model.encode(torch.from_numpy(rgb[unique_frames]).to(device))
            inverse = torch.from_numpy(inverse).to(device)
            batch, slots = batch_windows.shape
            logits = model.head(features[inverse].reshape(
                batch, slots * features.shape[1], *features.shape[-2:]))
            scores = readout_scores(logits, grid_hw)
            predicted_cells.append(scores["predicted_cell"].cpu().numpy())
            for name in SCORE_NAMES:
                score_chunks[name].append(scores[name].cpu().numpy())
            if start + len(ids) == len(val_indices) or (start // batch_size + 1) % 50 == 0:
                print(json.dumps({"processed": start + len(ids), "total": len(val_indices),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)

    predicted_cells = np.concatenate(predicted_cells)
    score_arrays = {name: np.concatenate(chunks) for name, chunks in score_chunks.items()}
    new_xy = grid_to_original(predicted_cells, grid_hw)
    if not np.array_equal(new_xy, saved_xy):
        raise ValueError(f"新forward与val_predictions位置不完全相等: {np.sum(np.any(new_xy != saved_xy, axis=1))}行")
    q_max_abs_difference = float(np.max(np.abs(score_arrays["q"] - saved_q)))
    if not np.allclose(score_arrays["q"], saved_q, rtol=1e-5, atol=1e-5):
        raise ValueError(f"新forward与val_predictions q不一致: max_abs={q_max_abs_difference}")
    score_arrays["q"] = saved_q
    score_arrays["q_times_m16"] = saved_q.astype(np.float32) * score_arrays["m16"]

    visibility = np.asarray([row["visibility_raw"] for row in saved_rows])
    located = visibility != 0
    target_xy = np.asarray([[row["x_raw"], row["y_raw"]] for row in saved_rows], dtype=float)
    error = np.linalg.norm(saved_xy - target_xy, axis=1)
    within16 = error <= 16
    emitted = saved_q >= .5
    event = located & within16
    categories = {
        "correct_location_output": located & within16 & emitted,
        "wrong_location_output": located & ~within16 & emitted,
        "correct_but_rejected": located & within16 & ~emitted,
        "wrong_and_rejected": located & ~within16 & ~emitted,
        "vc0_false_positive": ~located & emitted,
        "vc0_rejected": ~located & ~emitted,
    }
    category_summary = {name: score_summary(score_arrays, mask)
                        for name, mask in categories.items()}

    clip_ids = np.asarray([f'{row["game"]}/{row["clip"]}' for row in saved_rows])
    by_clip = {}
    for clip in sorted(set(clip_ids)):
        mask = clip_ids == clip
        by_clip[clip] = {
            "n": int(mask.sum()),
            "event_positive": int(np.sum(event & mask)),
            "emitted": int(np.sum(emitted & mask)),
            "auroc_q": auroc(score_arrays["q"][mask], event[mask]),
            "auroc_q_times_m16": auroc(score_arrays["q_times_m16"][mask], event[mask]),
            "event_scores": score_summary(score_arrays, mask & event),
            "non_event_scores": score_summary(score_arrays, mask & ~event),
        }

    scoreboard = (saved_xy[:, 0] == 54.5) & (saved_xy[:, 1] == 664.5)
    elapsed = time.perf_counter() - started
    report = {
        "scope": {
            "run": str(run.resolve()),
            "config": str((run / "config.json").resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "saved_predictions": str((run / "val_predictions.csv").resolve()),
            "rgb_cache": str(rgb_cache.resolve()),
            "checkpoint_epoch": checkpoint_epoch,
            "device": torch.cuda.get_device_name(),
            "dtype": "float32",
            "batch_size": batch_size,
            "torch_version": str(torch.__version__),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "temporal_input": temporal_input,
            "model_input_slots": input_slots,
            "eligibility_time_range": "original causal [t-2,t-1,t] windows; target t",
            "score_note": "q, conditional spatial mass, and their product are diagnostic scores, not calibrated probabilities",
            "neighborhood": "3x3 conditional-location cells centered on predicted argmax; equals original-center distance <=16px for the 10px grid",
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "validation_consistency": {
            "n": len(saved_rows),
            "identity_order_labels_equal": True,
            "predicted_xy_exactly_equal": True,
            "q_rtol": 1e-5,
            "q_atol": 1e-5,
            "q_max_abs_difference": q_max_abs_difference,
        },
        "categories": category_summary,
        "category_counts": {name: int(mask.sum()) for name, mask in categories.items()},
        "scoreboard_cell_54_5_664_5": {
            **score_summary(score_arrays, scoreboard),
            "post_hoc_group": True,
            "emitted": int(np.sum(scoreboard & emitted)),
            "rejected": int(np.sum(scoreboard & ~emitted)),
            "event_positive": int(np.sum(scoreboard & event)),
        },
        "auroc": {
            "all": {"q": auroc(score_arrays["q"], event),
                    "q_times_m16": auroc(score_arrays["q_times_m16"], event)},
            "original_output_q_ge_0_5": {
                "n": int(emitted.sum()),
                "q": auroc(score_arrays["q"][emitted], event[emitted]),
                "q_times_m16": auroc(score_arrays["q_times_m16"][emitted], event[emitted]),
            },
        },
        "by_clip": by_clip,
        "elapsed_seconds": elapsed,
        "timing_scope": "RGB mmap indexing + GPU float32 prefix/head forward + CPU score aggregation and saved-prediction checks",
    }

    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["game", "clip", "original_frame_id", "visibility_raw", "x_raw", "y_raw",
                         "pred_x", "pred_y", *SCORE_NAMES])
        for index, row in enumerate(saved_rows):
            writer.writerow([row[key] for key in ("game", "clip", "original_frame_id",
                                                   "visibility_raw", "x_raw", "y_raw")]
                            + [float(saved_xy[index, 0]), float(saved_xy[index, 1])]
                            + [float(score_arrays[name][index]) for name in SCORE_NAMES])
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output_json), "n": len(saved_rows),
                      "elapsed_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
