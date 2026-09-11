"""在已选完整 DINO checkpoint 上测量 game7 的 GT-query 局部端点关系。"""
import argparse
from collections import defaultdict
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
from ballmotion.correspondence import endpoint_logits, native_endpoint_cells
from train_tennis_heatmap import build_dino_model


CSV_FIELDS = ("game", "clip", "original_frame_id", "history_original_frame_id", "delta",
              "current_native_cell", "history_native_cell", "target_offset_index",
              "predicted_offset_index", "exact", "nll", "same_cell",
              "valid_candidate_count")


def rows_from_pairs(pairs, batch_ids, windows, frames, cells):
    """把 endpoint_logits 的批内索引恢复成原窗口身份。"""
    rows = []
    for delta, pair in enumerate(pairs, 1):
        if not len(pair["target"]):
            continue
        nll = F.cross_entropy(pair["logits"], pair["target"], reduction="none").cpu()
        predicted = pair["logits"].argmax(1).cpu()
        target = pair["target"].cpu()
        batch_indices = pair["batch_indices"].cpu()
        same_cell = pair["same_cell"].cpu()
        valid_counts = torch.isfinite(pair["logits"]).sum(1).cpu()
        for position, local_index in enumerate(batch_indices.tolist()):
            window_index = int(batch_ids[local_index])
            window = windows[window_index]
            current_frame = frames[int(window[2])]
            history_frame = frames[int(window[2 - delta])]
            if (current_frame["game"], current_frame["clip"]) != (
                    history_frame["game"], history_frame["clip"]):
                raise ValueError("端点跨越 clip，缓存窗口身份非法")
            rows.append({
                "game": current_frame["game"],
                "clip": current_frame["clip"],
                "original_frame_id": current_frame["original_frame_id"],
                "history_original_frame_id": history_frame["original_frame_id"],
                "delta": delta,
                "current_native_cell": int(cells[window_index, 2]),
                "history_native_cell": int(cells[window_index, 2 - delta]),
                "target_offset_index": int(target[position]),
                "predicted_offset_index": int(predicted[position]),
                "exact": bool(predicted[position] == target[position]),
                "nll": float(nll[position]),
                "same_cell": bool(same_cell[position]),
                "valid_candidate_count": int(valid_counts[position]),
            })
    return rows


def summarize(rows):
    return {
        "count": len(rows),
        "exact_count": sum(row["exact"] for row in rows),
        "r1": (sum(row["exact"] for row in rows) / len(rows) if rows else None),
        "nll": (sum(row["nll"] for row in rows) / len(rows) if rows else None),
    }


def aggregate(rows):
    identities = [(row["game"], row["clip"], str(row["original_frame_id"]), row["delta"])
                  for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("同一窗口端点身份重复")
    output = {}
    for delta in (1, 2):
        delta_rows = [row for row in rows if row["delta"] == delta]
        clips = defaultdict(list)
        for row in delta_rows:
            clips[f'{row["game"]}/{row["clip"]}'].append(row)
        output[str(delta)] = {
            "all": summarize(delta_rows),
            "same_cell": summarize([row for row in delta_rows if row["same_cell"]]),
            "moved": summarize([row for row in delta_rows if not row["same_cell"]]),
            "by_clip": {
                clip: {
                    "all": summarize(group),
                    "same_cell": summarize([row for row in group if row["same_cell"]]),
                    "moved": summarize([row for row in group if not row["same_cell"]]),
                }
                for clip, group in sorted(clips.items())
            },
        }
    macro = {}
    for group in ("all", "moved"):
        parts = [output[str(delta)][group] for delta in (1, 2)]
        macro[group] = {
            "count": sum(part["count"] for part in parts),
            "exact_count": sum(part["exact_count"] for part in parts),
            "r1": (sum(part["r1"] for part in parts) / 2
                   if all(part["r1"] is not None for part in parts) else None),
            "nll": (sum(part["nll"] for part in parts) / 2
                    if all(part["nll"] is not None for part in parts) else None),
            "aggregation": "equal mean of delta1 and delta2; counts are descriptive, not independent samples",
        }
    return output, macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = ROOT / args.run
    output_csv = run / "endpoint_relations.csv"
    output_json = run / "endpoint_relations.json"
    if output_csv.exists() or output_json.exists():
        raise ValueError("endpoint relation 输出已存在；拒绝覆盖")

    config = json.loads((run / "config.json").read_text())
    results = json.loads((run / "results.json").read_text())
    if config.get("model") != "dino" or config.get("temporal_input", "history") != "history":
        raise ValueError("诊断只支持完整 DINO 真实历史实验")
    if config.get("auxiliary", "none") not in ("none", "relation", "appearance"):
        raise ValueError("未知训练辅助组")
    if int(config.get("batch_size", 8)) != 8 or config.get("precision") != "float32; no AMP":
        raise ValueError("诊断要求正式 batch8 float32 实验配置")
    if (int(config.get("epochs", -1)), int(config.get("seed", -1)),
            int(config.get("train_frames", -1)), int(config.get("val_frames", -1))) != (
                30, 0, 12167, 1863):
        raise ValueError("诊断只支持 seed0、30 epoch 的完整 games1–7 实验")

    cache = ROOT / config["rgb_cache"]
    metadata = json.loads((cache / "metadata.json").read_text())
    expected_cache = {"games": list(range(1, 8)), "target_step": 1, "history": 2,
                      "input_hw": [288, 512], "resize": "PIL bilinear RGB"}
    if any(metadata.get("config", {}).get(key) != value for key, value in expected_cache.items()):
        raise ValueError("需要 games1–7、真实 [t-2,t-1,t] 的 512x288 RGB 缓存")
    frames = metadata["frames"]
    windows = np.asarray(metadata["windows"], dtype=np.int64)
    if windows.ndim != 2 or windows.shape[1] != 3:
        raise ValueError("诊断需要三帧真实窗口")
    val_ids = np.asarray([i for i, window in enumerate(windows)
                          if frames[int(window[2])]["game"] == "game7"], dtype=np.int64)
    if not len(val_ids):
        raise ValueError("RGB 缓存没有 game7 验证窗口")
    cells = native_endpoint_cells(frames, windows)
    rgb = np.load(cache / "rgb.npy", mmap_mode="r")
    if rgb.shape != (len(frames), 3, 288, 512) or rgb.dtype != np.uint8:
        raise ValueError("RGB 缓存必须为 N×3×288×512 uint8")

    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_epoch = int(checkpoint["epoch"])
    if checkpoint_epoch != int(results["best_epoch"]):
        raise ValueError("best.pt epoch 与主 F1 选中的 results best_epoch 不同")
    weights = ROOT / config["weights"]
    model = build_dino_model(weights)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()
    del checkpoint

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(val_ids), 8):
            ids = val_ids[start:start + 8]
            batch_windows = windows[ids]
            unique_frames, inverse = np.unique(batch_windows, return_inverse=True)
            pixels = torch.from_numpy(rgb[unique_frames]).to(device)
            unique_features = model.encode(pixels)
            inverse = torch.from_numpy(inverse).to(device)
            features = unique_features[inverse].reshape(len(ids), 3, 192, 36, 64)
            pairs = endpoint_logits(features, cells[ids].to(device), temperature=.1)
            rows.extend(rows_from_pairs(pairs, ids, windows, frames, cells))
            if start + len(ids) == len(val_ids) or (start // 8 + 1) % 50 == 0:
                print(json.dumps({"processed_windows": start + len(ids),
                                  "total_windows": len(val_ids), "eligible_pairs": len(rows),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    torch.cuda.synchronize()
    if not rows or {row["delta"] for row in rows} != {1, 2}:
        raise ValueError("game7 缺少某个 delta 的双 VC1、in-range pair")
    by_delta, macro = aggregate(rows)
    expected = {"1": {"all": 1518, "same_cell": 407, "moved": 1111},
                "2": {"all": 1491, "same_cell": 141, "moved": 1350}}
    actual = {delta: {group: by_delta[delta][group]["count"]
                      for group in ("all", "same_cell", "moved")}
              for delta in ("1", "2")}
    if actual != expected:
        raise ValueError(f"game7 合法 pair 几何计数不符: {actual} != {expected}")
    if any(not math.isfinite(row["nll"]) for row in rows):
        raise ValueError("target NLL 出现非有限值")
    elapsed = time.perf_counter() - started

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    diagnostic_status = subprocess.check_output(
        ["git", "status", "--short", "--", str(Path(__file__).relative_to(ROOT))],
        cwd=ROOT, text=True).strip() or "clean"
    report = {
        "scope": {
            "run": str(run.resolve()),
            "config": str((run / "config.json").resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_selection": config["selection"],
            "training_auxiliary": config.get("auxiliary", "none"),
            "training_code_revision": config["code_revision"],
            "diagnostic_code_revision": revision,
            "diagnostic_code_status": diagnostic_status,
            "rgb_cache": str(cache.resolve()),
            "rgb_array": str((cache / "rgb.npy").resolve()),
            "game": "game7",
            "batch_size": 8,
            "device": torch.cuda.get_device_name(),
            "precision": "float32 eval; no AMP",
            "temperature": .1,
            "candidate_support": {"delta1_radius2": 25, "delta2_radius4": 81},
            "input_time": "original causal [t-2,t-1,t] windows; target/query at t",
            "input_position": "current original VC1 GT center mapped to native 36x64 cell",
            "eligibility": "both endpoints original VC1 and history native cell in fixed local range",
            "query": "current-instance spatial-mean-centered, channel-L2-normalized feature for every arm",
            "appearance_note": "training-only appearance query w is not loaded or used",
        },
        "counts": {"validation_windows": len(val_ids), "eligible_pairs": len(rows),
                   "observed_eligible_by_delta": actual},
        "by_delta": by_delta,
        "macro_equal_delta": macro,
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "timing_scope": "RGB mmap indexing + H2D + GPU float32 prefix/local relation + per-pair CPU rows + final grouping; excludes imports, metadata/checkpoint/model loading and output writes",
    }
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")
    print(json.dumps({"output": str(output_json), "eligible_pairs": len(rows),
                      "elapsed_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
