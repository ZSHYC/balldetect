"""在缓存特征上训练定位读出，支持当前帧和严格因果的时间拼接。"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.tennis import grid_targets, grid_to_original
from ballmotion.probe import SpatialProbe, evaluate_predictions, frame_batch


def predict(model, array, windows, indices, batch_size, device, temporal_input, cost_array=None):
    model.eval()
    positions, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            selected = windows[ids]
            extra = cost_array[ids] if cost_array is not None else None
            x = torch.from_numpy(frame_batch(array, selected, temporal_input, extra)).to(device).float()
            logits = model(x)
            positions.extend(logits[:, :-1].argmax(1).cpu().tolist())
            probabilities.extend((1 - logits.softmax(1)[:, -1]).cpu().tolist())
    grid_hw = tuple(s * model.upscale for s in array.shape[-2:])
    return grid_to_original(positions, grid_hw), np.array(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(4), required=True)
    parser.add_argument("--output-stride", type=int, choices=(4, 8, 16, 32), help="默认原生网格；更细网格使用子格线性读出")
    parser.add_argument("--hidden-channels", type=int, default=0, help="0 为线性头；正数使用 1x1-GELU-3x3 读出")
    parser.add_argument("--temporal-input", choices=("current", "stack", "repeat"), default="current")
    parser.add_argument("--cost-cache", type=Path, help="同源窗口的局部对应缓存，仅与stack对照融合")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.hidden_channels < 0:
        raise ValueError("hidden_channels must be nonnegative")
    native_stride = 4 * 2 ** args.stage
    output_stride = args.output_stride or native_stride
    if output_stride > native_stride:
        raise ValueError("此读出仅支持原生或更细网格")
    upscale = native_stride // output_stride
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "history.jsonl").exists():
        raise ValueError("此运行目录已有实验；新实验请用新目录，避免覆盖旧结果")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    meta = json.loads((args.cache / "metadata.json").read_text())
    all_rows = meta["frames"]
    if set(r["game"] for r in all_rows) - {f"game{i}" for i in range(1, 8)}:
        raise ValueError("开发探针缓存不能包含最终测试比赛")
    windows = np.asarray(meta["windows"], dtype=np.int64) if "windows" in meta else np.arange(len(all_rows))[:, None]
    if args.temporal_input != "current" and windows.shape[1] < 2:
        raise ValueError("时间对照需要真实连续窗口缓存，不能把稀疏缓存相邻行充当连续帧")
    rows = [all_rows[i] for i in windows[:, -1]]
    num_frames = 1 if args.temporal_input == "current" else windows.shape[1]
    train_idx = np.array([i for i, r in enumerate(rows) if r["game"] != "game7"])
    val_idx = np.array([i for i, r in enumerate(rows) if r["game"] == "game7"])
    if not len(train_idx) or not len(val_idx):
        raise ValueError("train/val 必须非空")
    array = np.load(args.cache / f"stage{args.stage}.npy", mmap_mode="r")
    if list(array.shape) != meta["shapes"][args.stage]:
        raise ValueError("特征 shape 与帧元信息不匹配")
    cost_array, cost_config = None, None
    if args.cost_cache is not None:
        if args.temporal_input != "stack" or args.stage != 1:
            raise ValueError("本轮对应基线固定在stage1三帧appearance上")
        cost_config = json.loads((args.cost_cache / "metadata.json").read_text())
        if cost_config["source_cache"] != str(args.cache.resolve()) or cost_config["source_config"] != meta["config"]:
            raise ValueError("对应缓存与appearance缓存的来源条件不同")
        cost_array = np.load(args.cost_cache / "cost.npy", mmap_mode="r")
        if cost_array.shape != (len(windows), 106, *array.shape[-2:]):
            raise ValueError("对应缓存的目标窗口数/空间尺寸不同")
    target_xy = np.array([[r["x_raw"], r["y_raw"]] for r in rows], dtype=float)
    present = np.array([r["visibility_raw"] != 0 for r in rows])
    grid_hw = tuple(s * upscale for s in array.shape[-2:])
    targets = grid_targets(target_xy, present, grid_hw)
    appearance_channels = array.shape[1] * num_frames
    channels = appearance_channels + (cost_array.shape[1] if cost_array is not None else 0)
    model = SpatialProbe(channels, upscale=upscale, hidden_channels=args.hidden_channels,
                         num_frames=num_frames, appearance_channels=appearance_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    val_rows = [rows[i] for i in val_idx]
    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cache_config": meta["config"], "code_revision": code_revision,
              "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
              "train_frames": len(train_idx), "val_frames": len(val_idx),
              "parameters": sum(p.numel() for p in model.parameters()),
              "head": "framewise appearance GroupNorm + " + ("nonlinear" if args.hidden_channels else "linear")
                      + " subcell spatial; appearance-only linear absence"
                      + ("; raw-scale cost enters spatial only" if cost_array is not None else ""),
              "output_grid_hw": grid_hw, "upscale": upscale, "num_frames": num_frames,
              "target_slot": windows.shape[1] - 1,
              "appearance_channels": appearance_channels, "cost_config": cost_config,
              "selection": "maximum val conditional PCK@16; then PCK@8; first on ties"}
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(config, ensure_ascii=False), flush=True)
    best, best_epoch = (-1., -1.), 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        loss_sum = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            extra = cost_array[ids] if cost_array is not None else None
            x = torch.from_numpy(frame_batch(array, windows[ids], args.temporal_input, extra)).to(device).float()
            y = torch.from_numpy(targets[ids]).to(device)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        xy, prob = predict(model, array, windows, val_idx, args.batch_size, device, args.temporal_input, cost_array)
        metrics = evaluate_predictions(val_rows, xy, prob)
        score = (metrics["location"]["pck16"], metrics["location"]["pck8"])
        record = dict(epoch=epoch, train_loss=loss_sum / len(order), val=metrics["location"],
                      elapsed_seconds=time.perf_counter() - started)
        with (args.output / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(dict(model=model.state_dict(), epoch=epoch), args.output / "best.pt")
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=True)["model"])
    results = {}
    for split, ids in (("train", train_idx), ("val", val_idx)):
        xy, prob = predict(model, array, windows, ids, args.batch_size, device, args.temporal_input, cost_array)
        split_rows = [rows[i] for i in ids]
        results[split] = evaluate_predictions(split_rows, xy, prob)
        oracle_idx = np.minimum(targets[ids], grid_hw[0] * grid_hw[1] - 1)
        oracle_xy = grid_to_original(oracle_idx, grid_hw)
        results[split]["grid_oracle"] = evaluate_predictions(split_rows, oracle_xy, present[ids].astype(float))["location"]
        with (args.output / f"{split}_predictions.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["game", "clip", "original_frame_id", "visibility_raw", "x_raw", "y_raw",
                             "pred_x", "pred_y", "presence_probability"])
            for row, p, q in zip(split_rows, xy, prob):
                writer.writerow([row[k] for k in ("game", "clip", "original_frame_id", "visibility_raw", "x_raw", "y_raw")]
                                + [float(p[0]), float(p[1]), float(q)])
    results.update(best_epoch=best_epoch, elapsed_seconds=time.perf_counter() - started,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "val": results["val"]["location"],
                      "grid_oracle": results["val"]["grid_oracle"]}), flush=True)


if __name__ == "__main__":
    main()
