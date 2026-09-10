"""以相同 DINOv3 stage1 初始化比较冻结前缀与端到端微调。"""
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
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
from ballmotion.backbone_probe import BackboneProbe
from ballmotion.probe import SpatialProbe, evaluate_predictions
from ballmotion.tennis import grid_targets, grid_to_original
from dinov3.models.convnext import ConvNeXt


def load_initial_model(init_run, train_backbone):
    init_run = Path(init_run)
    config = json.loads((init_run / "config.json").read_text())
    if (config.get("stage") != 1 or config.get("temporal_input") != "stack"
            or config.get("num_frames") != 3 or config.get("target_slot") != 2
            or config.get("cost_cache") is not None or config.get("cost_config") is not None):
        raise ValueError("初始化仅接受 stage1、三帧 stack、无 cost 的现有读出")
    cache_config = config["cache_config"]
    if cache_config.get("backbone") != "dinov3_convnext_tiny":
        raise ValueError("初始化仅接受 DINOv3 ConvNeXt-Tiny")
    weights = Path(cache_config["weights"])
    weights = weights if weights.is_absolute() else ROOT / weights
    backbone = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True, mmap=True), strict=True)
    prefix = torch.nn.Sequential(backbone.downsample_layers[0], backbone.stages[0],
                                 backbone.downsample_layers[1], backbone.stages[1])
    output_stride = config.get("output_stride") or 8
    if output_stride not in (4, 8):
        raise ValueError("stage1 初始化输出 stride 不受支持")
    upscale = 8 // output_stride
    head = SpatialProbe(192 * 3, upscale=upscale,
                        hidden_channels=config.get("hidden_channels", 0), num_frames=3,
                        appearance_channels=192 * 3)
    checkpoint = torch.load(init_run / "best.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(checkpoint["model"], strict=True)
    config = dict(config)
    config["initial_epoch"] = int(checkpoint["epoch"])
    return BackboneProbe(prefix, head, train_backbone), config


def predict(model, rgb, windows, indices, batch_size, device, grid_hw):
    model.eval()
    positions, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            pixels = torch.from_numpy(rgb[windows[ids]]).to(device)
            logits = model(pixels)
            positions.extend(logits[:, :-1].argmax(1).cpu().tolist())
            probabilities.extend((1 - logits.softmax(1)[:, -1]).cpu().tolist())
    return grid_to_original(positions, grid_hw), np.asarray(probabilities)


def write_predictions(path, rows, xy, probability):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["game", "clip", "original_frame_id", "visibility_raw", "x_raw", "y_raw",
                         "pred_x", "pred_y", "presence_probability"])
        for row, point, presence in zip(rows, xy, probability):
            writer.writerow([row[k] for k in ("game", "clip", "original_frame_id",
                                               "visibility_raw", "x_raw", "y_raw")]
                            + [float(point[0]), float(point[1]), float(presence)])


def compare_initial(path, rows, xy, probability):
    with Path(path).open(newline="") as handle:
        reference = list(csv.DictReader(handle))
    identity = [(str(r["game"]), str(r["clip"]), str(r["original_frame_id"])) for r in rows]
    old_identity = [(r["game"], r["clip"], r["original_frame_id"]) for r in reference]
    old_xy = np.asarray([[float(r["pred_x"]), float(r["pred_y"])] for r in reference])
    old_presence = np.asarray([float(r["presence_probability"]) >= .5 for r in reference])
    comparison = {
        "rows": len(rows),
        "reference_rows": len(reference),
        "identity_order_equal": identity == old_identity,
        "coordinate_mismatches": (int(np.sum(np.any(np.asarray(xy) != old_xy, axis=1)))
                                    if len(rows) == len(reference) else None),
        "presence_decision_mismatches": (int(np.sum((np.asarray(probability) >= .5) != old_presence))
                                          if len(rows) == len(reference) else None),
    }
    comparison["passed"] = (comparison["identity_order_equal"]
                            and comparison["coordinate_mismatches"] == 0
                            and comparison["presence_decision_mismatches"] == 0)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--init-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-mode", choices=("frozen", "finetune"), required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=.0003)
    parser.add_argument("--backbone-lr", type=float, default=.00001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.output / "config.json").exists():
        raise ValueError("此目录已有实验配置；请用新目录避免覆盖")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    train_backbone = args.backbone_mode == "finetune"
    model, init_config = load_initial_model(args.init_run, train_backbone)

    rgb_meta = json.loads((args.rgb_cache / "metadata.json").read_text())
    init_cache = Path(init_config["cache"])
    init_cache = init_cache if init_cache.is_absolute() else ROOT / init_cache
    if (rgb_meta["source_cache"] != str(init_cache.resolve())
            or rgb_meta["source_config"] != init_config["cache_config"]):
        raise ValueError("RGB 缓存与初始化读出的目标或视觉条件不同")
    frames, windows = rgb_meta["frames"], np.asarray(rgb_meta["windows"], dtype=np.int64)
    if windows.ndim != 2 or windows.shape[1] != 3:
        raise ValueError("适配实验需要真实三帧窗口")
    if set(row["game"] for row in frames) - {f"game{i}" for i in range(1, 8)}:
        raise ValueError("适配缓存不能包含最终测试比赛")
    rows = [frames[i] for i in windows[:, -1]]
    train_idx = np.asarray([i for i, row in enumerate(rows) if row["game"] != "game7"])
    val_idx = np.asarray([i for i, row in enumerate(rows) if row["game"] == "game7"])
    if not len(train_idx) or not len(val_idx):
        raise ValueError("train/val 必须非空")
    rgb = np.load(args.rgb_cache / "rgb.npy", mmap_mode="r")
    expected_shape = (len(frames), 3, *rgb_meta["input_hw"])
    if rgb.shape != expected_shape or rgb.dtype != np.uint8 or list(rgb.shape) != rgb_meta["shape"]:
        raise ValueError("RGB 数组 shape/dtype 与元信息不符")

    model = model.to(device)
    upscale = model.head.upscale
    grid_hw = (rgb.shape[-2] // 8 * upscale, rgb.shape[-1] // 8 * upscale)
    target_xy = np.asarray([[r["x_raw"], r["y_raw"]] for r in rows], dtype=float)
    present = np.asarray([r["visibility_raw"] != 0 for r in rows])
    targets = grid_targets(target_xy, present, grid_hw)
    groups = [{"params": model.head.parameters(), "lr": args.head_lr}]
    if train_backbone:
        groups.append({"params": model.prefix.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    val_rows = [rows[i] for i in val_idx]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code_revision": revision,
        "init_run": str(args.init_run.resolve()),
        "initial_epoch": init_config["initial_epoch"],
        "rgb_source_cache": rgb_meta["source_cache"],
        "source_visual_config": rgb_meta["source_config"],
        "train_frames": len(train_idx), "val_frames": len(val_idx),
        "output_grid_hw": grid_hw, "target_slot": 2,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
        "precision": "float32; no AMP", "augmentation": None, "gradient_accumulation": None,
        "selection": "maximum val conditional PCK@16; then PCK@8; first on ties including epoch0",
        "timing_scope": "RGB mmap indexing + H2D + online prefix/head train and evaluation; excludes RGB cache creation",
    }
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(config, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    initial_xy, initial_probability = predict(model, rgb, windows, val_idx,
                                               args.batch_size, device, grid_hw)
    initial_metrics = evaluate_predictions(val_rows, initial_xy, initial_probability)
    comparison = compare_initial(args.init_run / "val_predictions.csv", val_rows,
                                 initial_xy, initial_probability)
    write_predictions(args.output / "initial_val_predictions.csv", val_rows,
                      initial_xy, initial_probability)
    initial = {"metrics": initial_metrics, "comparison": comparison}
    (args.output / "initial_val.json").write_text(json.dumps(initial, indent=2) + "\n")
    if not comparison["passed"]:
        raise ValueError(f"epoch0 未复现初始化验证预测: {comparison}")
    record = {"epoch": 0, "train_loss": None, "val": initial_metrics["location"],
              "epoch_seconds": time.perf_counter() - started,
              "elapsed_seconds": time.perf_counter() - started}
    with (args.output / "history.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    best = (initial_metrics["location"]["pck16"], initial_metrics["location"]["pck8"])
    best_epoch = 0
    torch.save({"model": model.state_dict(), "epoch": 0}, args.output / "best.pt")

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        model.prefix.eval()
        order = rng.permutation(train_idx)
        loss_sum = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            pixels = torch.from_numpy(rgb[windows[ids]]).to(device)
            target = torch.from_numpy(targets[ids]).to(device)
            loss = F.cross_entropy(model(pixels), target)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        xy, probability = predict(model, rgb, windows, val_idx, args.batch_size, device, grid_hw)
        metrics = evaluate_predictions(val_rows, xy, probability)
        score = (metrics["location"]["pck16"], metrics["location"]["pck8"])
        record = {"epoch": epoch, "train_loss": loss_sum / len(order),
                  "val": metrics["location"], "epoch_seconds": time.perf_counter() - epoch_started,
                  "elapsed_seconds": time.perf_counter() - started}
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output / "best.pt")

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    results = {"initial_val": initial_metrics, "initial_comparison": comparison}
    for split, ids in (("train", train_idx), ("val", val_idx)):
        xy, probability = predict(model, rgb, windows, ids, args.batch_size, device, grid_hw)
        split_rows = [rows[i] for i in ids]
        results[split] = evaluate_predictions(split_rows, xy, probability)
        oracle = grid_to_original(np.minimum(targets[ids], grid_hw[0] * grid_hw[1] - 1), grid_hw)
        results[split]["grid_oracle"] = evaluate_predictions(
            split_rows, oracle, present[ids].astype(float))["location"]
        write_predictions(args.output / f"{split}_predictions.csv", split_rows, xy, probability)
    results.update(best_epoch=best_epoch, elapsed_seconds=time.perf_counter() - started,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "val": results["val"]["location"]}), flush=True)


if __name__ == "__main__":
    main()
