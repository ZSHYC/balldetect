"""比较 BlurBall 真实历史与重复当前输入的已保存验证预测。"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.blurball import continuous_windows, evaluate_blurball


RADII = (4, 8, 16)


def _identity(row):
    return (row["game"], row["clip"], int(row["original_frame_id"]))


def read_predictions(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["original_frame_id"] = int(row["original_frame_id"])
        row["visibility_raw"] = int(row["visibility_raw"])
        for key in ("x_raw", "y_raw", "pred_x", "pred_y", "presence_probability"):
            row[key] = float(row[key])
    xy = np.array([[row["pred_x"], row["pred_y"]] for row in rows], dtype=float).reshape(-1, 2)
    q = np.array([row["presence_probability"] for row in rows], dtype=float)
    return rows, xy, q


def _validate_saved_rows(target_rows, saved_rows, path):
    if [_identity(row) for row in saved_rows] != [_identity(row) for row in target_rows]:
        raise ValueError(f"{path} 的目标身份或顺序与真实历史窗口不同")
    saved_labels = [(row["visibility_raw"], row["x_raw"], row["y_raw"])
                    for row in saved_rows]
    target_labels = [(int(row["visibility_raw"]), float(row["x_raw"]), float(row["y_raw"]))
                     for row in target_rows]
    if saved_labels != target_labels:
        raise ValueError(f"{path} 的原标签与RGB metadata不同")


def paired_raw(rows, history_xy, repeat_xy, mask):
    target = np.array([[row["x_raw"], row["y_raw"]] for row in rows], dtype=float)
    visible = np.array([row["visibility_raw"] == 1 for row in rows])
    selected = np.asarray(mask, dtype=bool) & visible
    history_error = np.linalg.norm(np.asarray(history_xy) - target, axis=1)
    repeat_error = np.linalg.norm(np.asarray(repeat_xy) - target, axis=1)
    result = {}
    for radius in RADII:
        history_ok = history_error[selected] < radius
        repeat_ok = repeat_error[selected] < radius
        n = int(selected.sum())
        rescued = int(np.sum(~repeat_ok & history_ok))
        broken = int(np.sum(repeat_ok & ~history_ok))
        result[str(radius)] = {
            "n": n, "rescued_by_history": rescued, "broken_by_history": broken,
            "both_correct": int(np.sum(repeat_ok & history_ok)),
            "both_wrong": int(np.sum(~repeat_ok & ~history_ok)),
            "net_history_pck_change": (rescued - broken) / n if n else None,
        }
    return result


def paired_decisions(rows, history_xy, history_q, repeat_xy, repeat_q):
    """以各模型自己的坐标和q配对，不把位置变化与输出变化混为一项。"""
    visible = np.array([row["visibility_raw"] == 1 for row in rows])
    target = np.array([[row["x_raw"], row["y_raw"]] for row in rows], dtype=float)
    history_error = np.linalg.norm(history_xy[visible] - target[visible], axis=1)
    repeat_error = np.linalg.norm(repeat_xy[visible] - target[visible], axis=1)
    history_rejected = history_q < .5
    repeat_rejected = repeat_q < .5
    result = {
        "matrix_axes": {"rows": "repeat_current", "columns": "history"},
        "v1_state_order": ["correct_emitted", "wrong_emitted", "correct_rejected", "wrong_rejected"],
        "v0_state_order": ["emitted", "rejected"],
        "v0": np.bincount(
            2 * repeat_rejected[~visible] + history_rejected[~visible],
            minlength=4).reshape(2, 2).tolist(),
        "v1": {},
    }
    for radius in RADII:
        history_state = (history_error >= radius).astype(int) + 2 * history_rejected[visible]
        repeat_state = (repeat_error >= radius).astype(int) + 2 * repeat_rejected[visible]
        result["v1"][str(radius)] = np.bincount(
            4 * repeat_state + history_state, minlength=16).reshape(4, 4).tolist()
    return result


def _group(rows, history_xy, history_q, repeat_xy, repeat_q, mask):
    ids = np.flatnonzero(mask)
    selected_rows = [rows[index] for index in ids]
    return {
        "n_targets": len(ids),
        "history": evaluate_blurball(
            selected_rows, history_xy[ids], history_q[ids], grouped=False),
        "repeat": evaluate_blurball(
            selected_rows, repeat_xy[ids], repeat_q[ids], grouped=False),
        "paired_raw": paired_raw(rows, history_xy, repeat_xy, mask),
        "paired_decisions": paired_decisions(
            selected_rows, history_xy[ids], history_q[ids], repeat_xy[ids], repeat_q[ids]),
    }


def _group_masks(rows, frames, windows):
    n = len(rows)
    current_visible = np.array([row["visibility_raw"] == 1 for row in rows])
    history_far = np.array([frames[window[0]]["visibility_raw"] for window in windows])
    history_near = np.array([frames[window[1]]["visibility_raw"] for window in windows])
    games = np.array([row["game"] for row in rows])
    lengths = np.array([row["l_raw"] for row in rows], dtype=float)
    valid_length = current_visible & np.isfinite(lengths) & (lengths >= 0)
    masks = {"all": np.ones(n, dtype=bool)}
    masks.update({f"match/{game}": games == game for game in sorted(set(games))})
    masks.update({
        "half_length/l0": valid_length & (lengths == 0),
        "half_length/0_2": valid_length & (lengths > 0) & (lengths <= 2),
        "half_length/2_5": valid_length & (lengths > 2) & (lengths <= 5),
        "half_length/5_10": valid_length & (lengths > 5) & (lengths <= 10),
        "half_length/gt10": valid_length & (lengths > 10),
    })
    for current in (0, 1):
        for far in (0, 1):
            for near in (0, 1):
                masks[f"visibility/V{current}_h{far}{near}"] = (
                    (current_visible == bool(current)) & (history_far == far) & (history_near == near))

    current_xy = np.array([[row["x_raw"], row["y_raw"]] for row in rows], dtype=float)
    for delta, slot in ((1, 1), (2, 0)):
        old_visible = np.array([frames[window[slot]]["visibility_raw"] == 1
                                for window in windows])
        old_xy = np.array([[frames[window[slot]]["x_raw"], frames[window[slot]]["y_raw"]]
                           for window in windows], dtype=float)
        distance = np.linalg.norm(current_xy - old_xy, axis=1)
        eligible = current_visible & old_visible
        masks[f"displacement/d{delta}_lt4"] = eligible & (distance < 4)
        masks[f"displacement/d{delta}_4_16"] = eligible & (distance >= 4) & (distance < 16)
        masks[f"displacement/d{delta}_ge16"] = eligible & (distance >= 16)
    for game in sorted(set(games)):
        masks[f"gt10_match/{game}"] = valid_length & (lengths > 10) & (games == game)
    return masks


def compare(rows, frames, windows, decoders):
    windows = np.asarray(windows, dtype=np.int64)
    if windows.shape != (len(rows), 3):
        raise ValueError("真实历史窗口必须与目标逐行对应且含[t-2,t-1,t]")
    if [_identity(frames[window[-1]]) for window in windows] != [_identity(row) for row in rows]:
        raise ValueError("目标没有与真实历史窗口末帧对齐")
    if set(decoders) != {"argmax", "local_readout"}:
        raise ValueError("必须同时提供argmax与local_readout")
    for model in ("history", "repeat"):
        argmax_q = np.asarray(decoders["argmax"][model][1])
        local_q = np.asarray(decoders["local_readout"][model][1])
        if not np.array_equal(argmax_q, local_q):
            raise ValueError(f"{model}模型的argmax与local_readout q不同")

    masks = _group_masks(rows, frames, windows)
    result = {"target_count": len(rows), "true_history_slots": ["t-2", "t-1", "t"],
              "radii": {"unit": "original pixels", "rule": "strict <", "values": list(RADII)},
              "decoders": {}}
    for decoder, values in decoders.items():
        history_xy, history_q = map(np.asarray, values["history"])
        repeat_xy, repeat_q = map(np.asarray, values["repeat"])
        history_metrics = evaluate_blurball(rows, history_xy, history_q, grouped=False)
        repeat_metrics = evaluate_blurball(rows, repeat_xy, repeat_q, grouped=False)
        result["decoders"][decoder] = {
            "history": history_metrics, "repeat": repeat_metrics,
            "groups": {name: _group(rows, history_xy, history_q, repeat_xy, repeat_q, mask)
                       for name, mask in masks.items() if mask.any()},
        }
    return result


def _load_run(run, target_rows):
    if not (run / "results.json").is_file():
        raise ValueError(f"运行尚未完成，缺少 {run / 'results.json'}")
    paths = {"argmax": run / "val_predictions.csv",
             "local_readout": run / "local_readout/val_predictions.csv"}
    result = {}
    for decoder, path in paths.items():
        if not path.is_file():
            raise ValueError(f"运行尚未完成，缺少 {path}")
        saved_rows, xy, q = read_predictions(path)
        _validate_saved_rows(target_rows, saved_rows, path)
        result[decoder] = (xy, q)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"不覆盖已有比较结果: {args.output}")

    history_config = json.loads((args.history / "config.json").read_text())
    repeat_config = json.loads((args.repeat / "config.json").read_text())
    if history_config.get("temporal_input", "history") != "history":
        raise ValueError("--history 不是原生真实历史运行")
    if repeat_config.get("temporal_input") != "repeat_current":
        raise ValueError("--repeat 不是重复当前输入运行")
    history_cache = ROOT / history_config["rgb_cache"]
    repeat_cache = ROOT / repeat_config["rgb_cache"]
    if history_cache.resolve() != repeat_cache.resolve():
        raise ValueError("两个运行未使用同一RGB metadata")
    metadata = json.loads((history_cache / "metadata.json").read_text())
    frames = metadata["frames"]
    windows, _ = continuous_windows(
        frames, metadata["windows"], history_config["continuity_boundaries"])
    val_windows = np.asarray([window for window in windows
                              if frames[window[-1]]["split"] == "val"], dtype=np.int64)
    rows = [frames[window[-1]] for window in val_windows]
    if len(rows) != int(history_config["val_targets"]):
        raise ValueError("历史运行的验证目标数与真实窗口不同")
    if int(repeat_config["val_targets"]) != len(rows):
        raise ValueError("repeat运行的验证目标数与历史运行不同")

    history = _load_run(args.history, rows)
    repeat = _load_run(args.repeat, rows)
    decoders = {decoder: {"history": history[decoder], "repeat": repeat[decoder]}
                for decoder in ("argmax", "local_readout")}
    result = compare(rows, frames, val_windows, decoders)
    result.update({
        "protocol": "blurball-full-temporal-control-v1",
        "history_run": str(args.history.resolve()), "repeat_run": str(args.repeat.resolve()),
        "rgb_metadata": str((history_cache / "metadata.json").resolve()),
        "scope": "validation match18-21 only; GT centers/visibility/l used for post-hoc analysis",
        "q_contract": "each model retains its own q; argmax and local_readout share q within model",
        "paired_direction": "repeat_current baseline -> true history challenger",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({decoder: result["decoders"][decoder]["groups"]["all"]["paired_raw"]
                      for decoder in ("argmax", "local_readout")}, indent=2))


if __name__ == "__main__":
    main()
