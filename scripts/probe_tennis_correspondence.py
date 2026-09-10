"""用当前帧 GT cell 的 oracle query 诊断冻结特征跨帧对应。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.tennis import grid_targets, grid_to_original


def analyze_pair(current, history, query_cell, true_cell, target_xy, grid_hw,
                 original_hw=(720, 1280)):
    """返回一个 oracle query 对一个历史特征图的全局和局部 cosine 诊断。"""
    h, w = grid_hw
    if current.shape != history.shape or current.shape[-2:] != grid_hw:
        raise ValueError("Feature maps do not match the requested grid")
    if not (0 <= query_cell < h * w and 0 <= true_cell < h * w):
        raise ValueError("Cell index is outside the feature grid")
    target_xy = np.asarray(target_xy, dtype=float)
    if target_xy.shape != (2,) or not np.isfinite(target_xy).all():
        raise ValueError("Historical GT coordinate must be a finite (x, y) pair")
    qy, qx = divmod(query_cell, w)
    ty, tx = divmod(true_cell, w)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    distances = torch.maximum((yy - qy).abs(), (xx - qx).abs()).flatten()
    result = {}
    for name, centered in (("raw", False), ("spatial_mean_centered", True)):
        query_map, key_map = current.float(), history.float()
        if centered:
            query_map = query_map - query_map.mean((-2, -1), keepdim=True)
            key_map = key_map - key_map.mean((-2, -1), keepdim=True)
        query = F.normalize(query_map[:, qy, qx], dim=0)
        keys = F.normalize(key_map.flatten(1).T, dim=1)
        scores = keys @ query
        true_score = scores[true_cell]
        global_order = torch.argsort(scores, descending=True, stable=True)
        searches = {}
        for label, radius in (("global", None), ("radius2", 2), ("radius4", 4), ("radius8", 8)):
            candidates = torch.arange(h * w) if radius is None else torch.nonzero(
                distances <= radius, as_tuple=False).flatten()
            predicted = int(candidates[torch.argmax(scores[candidates])])
            predicted_xy = grid_to_original([predicted], grid_hw, original_hw)[0]
            error = float(np.linalg.norm(predicted_xy - target_xy))
            item = {
                "predicted_cell": predicted,
                "predicted_xy": predicted_xy.tolist(),
                "error_px": error,
                "pck16": error <= 16,
                "pck32": error <= 32,
                "geometric_coverage": radius is None or max(abs(ty - qy), abs(tx - qx)) <= radius,
            }
            if radius is None:
                rank = 1 + int(torch.sum(scores > true_score))
                margin = float(true_score - scores[query_cell])
                item.update({
                    "optimistic_true_cell_rank": rank,
                    **{f"recall_at_{k}": int(torch.any(global_order[:k] == true_cell))
                       for k in (1, 5, 10)},
                    "true_cell_score": float(true_score),
                    "same_coordinate_score": float(scores[query_cell]),
                    "true_minus_same_coordinate_score": margin,
                    "different_cells": true_cell != query_cell,
                })
            searches[label] = item
        result[name] = searches
    return result


def summarize(pairs):
    summary = {"n": len(pairs), "modes": {}}
    for mode in ("raw", "spatial_mean_centered"):
        mode_summary = {}
        for search in ("global", "radius2", "radius4", "radius8"):
            items = [pair["results"][mode][search] for pair in pairs]
            errors = np.array([item["error_px"] for item in items])
            values = {
                "n": len(items),
                "mean_error_px": float(errors.mean()) if len(errors) else None,
                "median_error_px": float(np.median(errors)) if len(errors) else None,
                "pck16": float(np.mean([item["pck16"] for item in items])) if items else None,
                "pck32": float(np.mean([item["pck32"] for item in items])) if items else None,
                "geometric_coverage_n": int(sum(item["geometric_coverage"] for item in items)),
                "geometric_coverage": float(np.mean([item["geometric_coverage"] for item in items]))
                if items else None,
            }
            if search == "global":
                ranks = np.array([item["optimistic_true_cell_rank"] for item in items])
                different = [item for item in items if item["different_cells"]]
                values.update({
                    "mean_optimistic_true_cell_rank": float(ranks.mean()) if len(ranks) else None,
                    "median_optimistic_true_cell_rank": float(np.median(ranks)) if len(ranks) else None,
                    **{f"recall_at_{k}": float(np.mean([item[f"recall_at_{k}"] for item in items]))
                       if items else None for k in (1, 5, 10)},
                    "true_minus_same_coordinate_score": {
                        "all_mean": float(np.mean([item["true_minus_same_coordinate_score"]
                                                   for item in items])) if items else None,
                        "different_cells_n": len(different),
                        "different_cells_mean": float(np.mean([
                            item["true_minus_same_coordinate_score"] for item in different
                        ])) if different else None,
                    },
                })
            mode_summary[search] = values
        summary["modes"][mode] = mode_summary
    return summary


def summarize_groups(pairs):
    clips = defaultdict(list)
    for pair in pairs:
        clips[f'{pair["game"]}/{pair["clip"]}'].append(pair)
    return {
        "all": summarize(pairs),
        "easy_both_endpoints": summarize([
            pair for pair in pairs
            if pair["current_visibility"] == pair["history_visibility"] == 1
        ]),
        "different_native_cells": summarize([
            pair for pair in pairs if pair["query_cell"] != pair["true_history_cell"]
        ]),
        "by_clip": {clip: summarize(group) for clip, group in sorted(clips.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()

    meta_path = args.cache / "metadata.json"
    feature_path = args.cache / "stage1.npy"
    if not meta_path.exists() or not feature_path.exists():
        raise ValueError("Cache metadata or stage1 feature file is missing")
    meta = json.loads(meta_path.read_text())
    frames = meta["frames"]
    windows = np.asarray(meta["windows"], dtype=np.int64)
    array = np.load(feature_path, mmap_mode="r")
    grid_hw = tuple(array.shape[-2:])
    if grid_hw != (36, 64):
        raise ValueError("This diagnostic is locked to the native 36x64 stage1 grid")
    if windows.ndim != 2 or windows.shape[1] < 3:
        raise ValueError("Cache does not contain the required causal h2 windows")
    if array.shape[0] != len(frames):
        raise ValueError("Feature and metadata frame counts differ")

    original_hw = (720, 1280)
    pairs = []
    candidate_windows = 0
    for window in windows:
        current_row = frames[int(window[-1])]
        if current_row["game"] != "game7":
            continue
        candidate_windows += 1
        if (current_row["height"], current_row["width"]) != original_hw:
            raise ValueError("This diagnostic only supports the published 1280x720 coordinates")
        current = (torch.from_numpy(np.array(array[int(window[-1])], dtype=np.float32, copy=True))
                   if current_row["label_state"] == "located" else None)
        for delta in (1, 2):
            history_row = frames[int(window[-1 - delta])]
            if (history_row["game"], history_row["clip"]) != (current_row["game"], current_row["clip"]):
                raise ValueError("Causal cache window crosses a game or clip boundary")
            if int(current_row["original_frame_id"]) - int(history_row["original_frame_id"]) != delta:
                raise ValueError("Causal cache window does not contain consecutive original frames")
            if current_row["label_state"] != "located" or history_row["label_state"] != "located":
                continue
            xy = np.array([[current_row["x_raw"], current_row["y_raw"]],
                           [history_row["x_raw"], history_row["y_raw"]]], dtype=float)
            cells = grid_targets(xy, [True, True], grid_hw, original_hw)
            current_yx, history_yx = divmod(int(cells[0]), grid_hw[1]), divmod(
                int(cells[1]), grid_hw[1])
            history = torch.from_numpy(np.array(array[int(window[-1 - delta])], dtype=np.float32, copy=True))
            pairs.append({
                "game": current_row["game"],
                "clip": current_row["clip"],
                "current_original_frame_id": int(current_row["original_frame_id"]),
                "history_original_frame_id": int(history_row["original_frame_id"]),
                "delta": delta,
                "current_visibility": current_row["visibility_raw"],
                "history_visibility": history_row["visibility_raw"],
                "current_gt_xy": xy[0].tolist(),
                "history_gt_xy": xy[1].tolist(),
                "displacement_px": float(np.linalg.norm(xy[0] - xy[1])),
                "native_displacement_xy": [history_yx[1] - current_yx[1],
                                             history_yx[0] - current_yx[0]],
                "native_displacement_chebyshev": max(abs(history_yx[0] - current_yx[0]),
                                                       abs(history_yx[1] - current_yx[1])),
                "query_cell": int(cells[0]),
                "true_history_cell": int(cells[1]),
                "results": analyze_pair(current, history, int(cells[0]), int(cells[1]), xy[1],
                                        grid_hw, original_hw),
            })

    aggregate = {}
    for delta in (1, 2):
        selected = [pair for pair in pairs if pair["delta"] == delta]
        aggregate[str(delta)] = summarize_groups(selected)

    output = {
        "scope": {
            "diagnostic": "frozen stage1 feature correspondence with a current-GT oracle query",
            "limitation": "Uses the known current ball cell; this is not automatic detection or a proposed motion module.",
            "data": "game7 target windows already present in the supplied causal h2 cache",
            "eligible_pairs": "current and history endpoints both have label_state=located; no training set is changed",
            "deltas": [1, 2],
            "grid_hw": list(grid_hw),
            "original_hw": list(original_hw),
            "search": "global and Chebyshev radii 2/4/8 centered on the current oracle query cell",
            "tie_rule": "optimistic rank is 1 + strictly higher scores; argmax and recall@k break score ties by lower cell index",
            "spatial_mean_centered": "subtract each frame/channel spatial mean before cosine; not camera-motion separation",
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "cache": str(args.cache.resolve()),
            "stage": 1,
        },
        "counts": {
            "game7_candidate_windows": candidate_windows,
            "eligible_pairs": {str(delta): sum(pair["delta"] == delta for pair in pairs)
                               for delta in (1, 2)},
        },
        "aggregate": aggregate,
        "pairs": pairs,
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "cache read + CPU matching + aggregation; excludes imports and final JSON write",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
