"""对完整 DINO 模型做固定预测原生块的四子格 GT oracle 几何诊断。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.probe import evaluate_predictions
from ballmotion.tennis import grid_targets, grid_to_original
from scripts.compare_predictions import read_predictions


OUTPUT_GRID_HW = (72, 128)
NATIVE_GRID_HW = (36, 64)
ORIGINAL_HW = (720, 1280)


def _fine_grid_indices(xy):
    """验证并恢复72x128输出格索引；训练预测必须落在10px格中心。"""
    indices = grid_targets(xy, np.ones(len(xy), dtype=bool), OUTPUT_GRID_HW)
    if np.any(indices < 0) or np.any(indices >= np.prod(OUTPUT_GRID_HW)):
        raise ValueError("预测坐标超出72x128输出格")
    recovered = grid_to_original(indices, OUTPUT_GRID_HW)
    if not np.allclose(recovered, xy, rtol=0, atol=1e-6):
        raise ValueError("预测坐标不在72x128的10px输出格中心")
    return indices // OUTPUT_GRID_HW[1], indices % OUTPUT_GRID_HW[1]


def fixed_block_subcell_oracle(predicted_xy, target_xy, located):
    """固定实际预测36x64块，在其四个72x128子格中选择离GT最近者。"""
    predicted_xy = np.asarray(predicted_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    located = np.asarray(located, dtype=bool)
    if predicted_xy.shape != target_xy.shape or predicted_xy.shape != (len(located), 2):
        raise ValueError("预测、GT与located长度不一致")
    if not np.isfinite(predicted_xy).all() or not np.isfinite(target_xy[located]).all():
        raise ValueError("预测或有位置GT含非有限值")

    fine_rows, fine_cols = _fine_grid_indices(predicted_xy)
    native_rows, native_cols = fine_rows // 2, fine_cols // 2
    predicted_blocks = native_rows * NATIVE_GRID_HW[1] + native_cols
    target_blocks = np.full(len(located), -1, dtype=np.int64)
    target_blocks[located] = grid_targets(
        target_xy[located], np.ones(np.sum(located), dtype=bool), NATIVE_GRID_HW)

    oracle_xy = predicted_xy.copy()
    ids = np.flatnonzero(located)
    if len(ids):
        child_rows = 2 * native_rows[ids, None] + np.array([0, 0, 1, 1])
        child_cols = 2 * native_cols[ids, None] + np.array([0, 1, 0, 1])
        child_indices = child_rows * OUTPUT_GRID_HW[1] + child_cols
        candidates = grid_to_original(child_indices, OUTPUT_GRID_HW)
        distances = np.linalg.norm(candidates - target_xy[ids, None, :], axis=2)
        oracle_xy[ids] = candidates[np.arange(len(ids)), distances.argmin(axis=1)]
    return oracle_xy, predicted_blocks, target_blocks


def _pck(error):
    return {"n": int(len(error)),
            **{f"pck{radius}": float(np.mean(error <= radius)) if len(error) else None
               for radius in (8, 16, 32)}}


def _group_summary(mask, actual_error, oracle_error, predicted_blocks, target_blocks):
    mask = np.asarray(mask, dtype=bool)
    actual_hit = mask & (actual_error <= 8)
    recoverable = mask & (actual_error > 8) & (oracle_error <= 8)
    no_child = mask & (oracle_error > 8)
    same_block = mask & (predicted_blocks == target_blocks)
    n = int(mask.sum())
    return {
        "actual": _pck(actual_error[mask]),
        "fixed_predicted_block_gt_oracle": _pck(oracle_error[mask]),
        "tolerance8_mutually_exclusive": {
            "actual_hit": int(actual_hit.sum()),
            "recoverable_within_selected_block": int(recoverable.sum()),
            "no_child_within_tolerance": int(no_child.sum()),
        },
        "predicted_native_block_equals_gt_native_block": {
            "count": int(same_block.sum()),
            "n": n,
            "fraction": float(same_block.sum() / n) if n else None,
        },
    }


def _check_saved_pck(split, actual_metrics, saved_metrics):
    checks = [("all", actual_metrics["location"], saved_metrics["location"])]
    checks += [(f"visibility{v}", actual_metrics["by_visibility"][str(v)],
                saved_metrics["by_visibility"][str(v)]) for v in (1, 2, 3)]
    checks += [(clip, metrics, saved_metrics["by_clip"][clip])
               for clip, metrics in actual_metrics["by_clip"].items()]
    for name, actual, saved in checks:
        if actual["n"] != saved["n"] or any(
                not np.isclose(actual[f"pck{radius}"], saved[f"pck{radius}"], rtol=0, atol=1e-12)
                for radius in (8, 16, 32) if actual[f"pck{radius}"] is not None):
            raise ValueError(f"{split}/{name} 未重现results.json中的PCK")


def analyze_split(run, split, saved_metrics):
    rows, predicted_xy, probability = read_predictions(run / f"{split}_predictions.csv")
    target_xy = np.asarray([[row["x_raw"], row["y_raw"]] for row in rows], dtype=float)
    visibility = np.asarray([row["visibility_raw"] for row in rows])
    located = visibility != 0
    oracle_xy, predicted_blocks, target_blocks = fixed_block_subcell_oracle(
        predicted_xy, target_xy, located)

    actual_metrics = evaluate_predictions(rows, predicted_xy, probability)
    oracle_metrics = evaluate_predictions(rows, oracle_xy, probability)
    _check_saved_pck(split, actual_metrics, saved_metrics)
    actual_error = np.linalg.norm(predicted_xy - target_xy, axis=1)
    oracle_error = np.linalg.norm(oracle_xy - target_xy, axis=1)
    if np.any(oracle_error[located] > actual_error[located] + 1e-12):
        raise ValueError(f"{split} oracle误差大于实际预测，块或子格映射错误")
    if not np.array_equal(oracle_xy[~located], predicted_xy[~located]):
        raise ValueError(f"{split} absence行坐标被改变")

    clips = np.asarray([row["game"] + "/" + row["clip"] for row in rows])
    groups = {"all": located,
              **{f"VC{v}": visibility == v for v in (1, 2, 3)},
              **{clip: located & (clips == clip) for clip in sorted(set(clips))}}
    return {
        "n_frames": len(rows),
        "n_located": int(located.sum()),
        "validation": {
            "saved_pck_reproduced": True,
            "all_predictions_on_72x128_grid": True,
            "oracle_error_never_exceeds_actual": True,
            "absence_coordinates_unchanged": True,
        },
        "groups": {name: _group_summary(mask, actual_error, oracle_error,
                                          predicted_blocks, target_blocks)
                   for name, mask in groups.items()},
        "fixed_probability_detection_gt_oracle": {
            "note": "presence_probability固定；位置由GT在预测块四子格内选择，属于GT oracle",
            "presence": oracle_metrics["presence"],
            **{f"detection{radius}": oracle_metrics[f"detection{radius}"]
               for radius in (8, 16, 32)},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    results = json.loads((args.run / "results.json").read_text())
    head = config.get("model_config", {}).get("head", {})
    if (config.get("model") != "dino" or config.get("output_grid_hw") != [72, 128]
            or head.get("upscale") != 2 or config["model_config"].get("prefix_output_channels") != 192):
        raise ValueError("该诊断只支持当前DINO stage1 36x64原生格到72x128四子格头")

    output = {
        "scope": ("GT几何诊断：固定每帧模型实际预测的36x64原生块，只在其四个真实72x128"
                  "子格中心中以GT选最近点；不改变训练预测或检查点选择"),
        "run": str(args.run),
        "best_epoch": results["best_epoch"],
        "geometry": {"original_hw": list(ORIGINAL_HW),
                     "native_grid_hw": list(NATIVE_GRID_HW),
                     "output_grid_hw": list(OUTPUT_GRID_HW),
                     "output_grid_spacing_original_px": 10,
                     "children_per_native_block": 4},
        "interpretation_limits": [
            "结果仅是固定已预测粗块后的四子格几何上限，不是可部署推理方法",
            "GT oracle提升不证明细位置信息能从模型特征中读出",
            "预测原生块等于GT原生块不等同于在给定像素容差内命中",
        ],
        "splits": {split: analyze_split(args.run, split, results[split])
                   for split in ("train", "val")},
    }
    path = args.run / "subcell_diagnostic.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    for split, summary in output["splits"].items():
        group = summary["groups"]["all"]
        print(split, json.dumps({"actual": group["actual"],
                                 "oracle": group["fixed_predicted_block_gt_oracle"],
                                 "counts8": group["tolerance8_mutually_exclusive"]},
                                ensure_ascii=False))


if __name__ == "__main__":
    main()
