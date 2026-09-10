"""训练 Tennis 全量因果 HRNet 或 DINOv3 共同任务基线。"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
from ballmotion.backbone_probe import BackboneProbe
from ballmotion.heatmap import disk_targets, heatmap_predictions, quality_focal_loss
from ballmotion.probe import SpatialProbe, evaluate_predictions
from ballmotion.tennis import grid_targets, grid_to_original
from dinov3.models.convnext import ConvNeXt
from third_party.wasb.hrnet import HRNet


def model_input(rgb, windows, ids, device, model_name):
    pixels = torch.from_numpy(rgb[windows[ids]]).to(device)
    if model_name == "dino":
        return pixels
    pixels = pixels.float().div_(255)
    for channel, (mean, std) in enumerate(zip((.485, .456, .406), (.229, .224, .225))):
        pixels[:, :, channel].sub_(mean).div_(std)
    return pixels.flatten(1, 2)


def predict(model, rgb, windows, indices, batch_size, device, model_name, grid_hw):
    model.eval()
    xy, confidence = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            logits = model(model_input(rgb, windows, ids, device, model_name))
            if model_name == "hrnet":
                batch_xy, batch_confidence = heatmap_predictions(logits[0])
            else:
                batch_xy = grid_to_original(logits[:, :-1].argmax(1).cpu().numpy(), grid_hw)
                batch_confidence = (1 - logits.softmax(1)[:, -1]).cpu().numpy()
            xy.append(batch_xy)
            confidence.append(batch_confidence)
    return np.concatenate(xy), np.concatenate(confidence)


def write_predictions(path, rows, xy, confidence):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["game", "clip", "original_frame_id", "visibility_raw", "x_raw", "y_raw",
                         "pred_x", "pred_y", "presence_probability"])
        for row, point, probability in zip(rows, xy, confidence):
            writer.writerow([row[k] for k in ("game", "clip", "original_frame_id",
                                               "visibility_raw", "x_raw", "y_raw")]
                            + [float(point[0]), float(point[1]), float(probability)])


def validation_record(metrics):
    return {
        "location": metrics["location"],
        "detection16": metrics["detection16"],
        "detection8": metrics["detection8"],
        "presence": metrics["presence"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=("hrnet", "dino"), default="hrnet")
    args = parser.parse_args()
    if (args.output / "config.json").exists():
        raise ValueError("此目录已有实验配置；请用新目录避免覆盖")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    rgb_meta = json.loads((args.rgb_cache / "metadata.json").read_text())
    expected_input = {"games": list(range(1, 8)), "target_step": 1, "history": 2,
                      "input_hw": [288, 512], "resize": "PIL bilinear RGB"}
    cache_config = rgb_meta.get("config", {})
    if any(cache_config.get(key) != value for key, value in expected_input.items()):
        raise ValueError("需要 games1–7、step1/history2、PIL 512×288 的全量 RGB 缓存")
    frames = rgb_meta["frames"]
    windows = np.asarray(rgb_meta["windows"], dtype=np.int64)
    if windows.ndim != 2 or windows.shape[1] != 3:
        raise ValueError("全量因果实验需要真实三帧窗口")
    if set(row["game"] for row in frames) - {f"game{i}" for i in range(1, 8)}:
        raise ValueError("全量因果缓存不能包含最终测试比赛")
    rows = [frames[i] for i in windows[:, -1]]
    train_idx = np.asarray([i for i, row in enumerate(rows) if row["game"] != "game7"])
    val_idx = np.asarray([i for i, row in enumerate(rows) if row["game"] == "game7"])
    if not len(train_idx) or not len(val_idx):
        raise ValueError("train/val 必须非空")

    rgb = np.load(args.rgb_cache / "rgb.npy", mmap_mode="r")
    input_hw = tuple(rgb_meta["input_hw"])
    if (rgb.shape != (len(frames), 3, *input_hw) or rgb.dtype != np.uint8
            or list(rgb.shape) != rgb_meta["shape"]):
        raise ValueError("RGB 数组 shape/dtype 与元信息不符")
    if input_hw != (288, 512):
        raise ValueError("RGB 输入尺寸必须为 288×512")

    if args.model == "hrnet":
        model_cfg = OmegaConf.load(ROOT / "configs/wasb_hrnet_causal.yaml")
        model = HRNet(model_cfg).to(device)
        grid_hw = (int(model_cfg.out_height), int(model_cfg.out_width))
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 1e-3,
                                       "name": "model"}], weight_decay=0)
        model_name = "WASB-HRNet common-task adaptation"
        model_config = OmegaConf.to_container(model_cfg, resolve=True)
        weights = None
        upstream_revision = "923462cacdeb3353b84ddebdedb3f4b7a8553b0f"
        initialization = "PyTorch default initialization; HRNet.init_weights not called"
        loss_config = {"name": "probability quality focal loss", "beta": 2,
                       "reduction": "mean over Bx1xHxW", "positive_disk_radius": 2.5}
        optimizer_config = {"name": "Adam", "weight_decay": 0,
                            "parameter_groups": {"model": {"lr": 1e-3}},
                            "lr_milestones_after_epoch": [10, 20], "lr_gamma": .1}
    else:
        weights_path = (ROOT / "models/pretrained/dinov3/lvd1689m/"
                        "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth")
        backbone = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True,
                                             mmap=True), strict=True)
        prefix = torch.nn.Sequential(backbone.downsample_layers[0], backbone.stages[0],
                                     backbone.downsample_layers[1], backbone.stages[1])
        del backbone
        head = SpatialProbe(576, upscale=2, hidden_channels=32, num_frames=3,
                            appearance_channels=576)
        model = BackboneProbe(prefix, head, train_backbone=True).to(device)
        grid_hw = (72, 128)
        optimizer = torch.optim.AdamW([
            {"params": model.head.parameters(), "lr": 3e-4, "name": "head"},
            {"params": model.prefix.parameters(), "lr": 1e-5, "name": "prefix"},
        ], weight_decay=.01)
        model_name = "DINOv3 ConvNeXt-Tiny stage1 + new three-frame SpatialProbe"
        model_config = {"backbone": "dinov3_convnext_tiny", "depths": [3, 3, 9, 3],
                        "dims": [96, 192, 384, 768], "prefix_output_channels": 192,
                        "head": {"input_channels": 576, "hidden_channels": 32,
                                 "num_frames": 3, "upscale": 2, "appearance_channels": 576}}
        weights = str(weights_path)
        upstream_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/dinov3", text=True).strip()
        initialization = "official DINOv3 backbone weights; newly initialized SpatialProbe head"
        loss_config = {"name": "cross_entropy", "classes": 72 * 128 + 1,
                       "absence_class": 72 * 128}
        optimizer_config = {"name": "AdamW", "weight_decay": .01,
                            "parameter_groups": {"head": {"lr": 3e-4},
                                                 "prefix": {"lr": 1e-5}},
                            "schedule": "constant"}

    target_xy = np.asarray([[r["x_raw"], r["y_raw"]] for r in rows], dtype=float)
    present = np.asarray([r["visibility_raw"] != 0 for r in rows])
    target_indices = grid_targets(target_xy, present, grid_hw)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "code_revision": revision,
        "model_name": model_name,
        "model_config": model_config,
        "weights": weights,
        "upstream_revision": upstream_revision,
        "initialization": initialization,
        "loss": loss_config,
        "cache_config": cache_config,
        "train_frames": len(train_idx),
        "val_frames": len(val_idx),
        "target_slot": 2,
        "output_grid_hw": grid_hw,
        "optimizer": optimizer_config,
        "selection": "maximum val detection F1@16; then F1@8; first on ties including epoch0",
        "seeds": {"torch": args.seed, "numpy": args.seed},
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "device": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "precision": "float32; no AMP",
        "batch_size": args.batch_size,
        "augmentation": None,
        "gradient_accumulation": None,
        "timing_scope": "RGB mmap indexing + H2D + online model train and evaluation; excludes RGB cache creation",
    }
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(config, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    val_rows = [rows[i] for i in val_idx]
    val_xy, val_confidence = predict(model, rgb, windows, val_idx, args.batch_size, device,
                                     args.model, grid_hw)
    initial_metrics = evaluate_predictions(val_rows, val_xy, val_confidence)
    record = {"epoch": 0, "train_loss": None, "val": validation_record(initial_metrics),
              "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
              "epoch_seconds": time.perf_counter() - started,
              "elapsed_seconds": time.perf_counter() - started}
    with (args.output / "history.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    best = (initial_metrics["detection16"]["f1"], initial_metrics["detection8"]["f1"])
    best_epoch = 0
    torch.save({"model": model.state_dict(), "epoch": 0}, args.output / "best.pt")

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        if args.model == "dino":
            model.prefix.eval()
        order = rng.permutation(train_idx)
        total_batches = (len(order) + args.batch_size - 1) // args.batch_size
        loss_sum = 0.
        for batch_number, start in enumerate(range(0, len(order), args.batch_size), 1):
            ids = order[start:start + args.batch_size]
            class_target = torch.from_numpy(target_indices[ids]).to(device)
            logits = model(model_input(rgb, windows, ids, device, args.model))
            if args.model == "hrnet":
                target = disk_targets(class_target, grid_hw)
                logits = logits[0]
                if logits.shape != target.shape:
                    raise ValueError(f"HRNet 输出 {tuple(logits.shape)} 与目标 {tuple(target.shape)} 不同")
                loss = quality_focal_loss(logits, target)
            else:
                loss = F.cross_entropy(logits, class_target)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            processed = start + len(ids)
            if batch_number % 200 == 0 or processed == len(order):
                print(json.dumps({"epoch": epoch, "batch": batch_number,
                                  "total_batches": total_batches,
                                  "train_loss_so_far": loss_sum / processed,
                                  "epoch_elapsed_seconds": time.perf_counter() - epoch_started}),
                      flush=True)

        val_xy, val_confidence = predict(model, rgb, windows, val_idx, args.batch_size,
                                         device, args.model, grid_hw)
        metrics = evaluate_predictions(val_rows, val_xy, val_confidence)
        score = (metrics["detection16"]["f1"], metrics["detection8"]["f1"])
        record = {"epoch": epoch, "train_loss": loss_sum / len(order),
                  "val": validation_record(metrics),
                  "learning_rates": {group["name"]: group["lr"]
                                     for group in optimizer.param_groups},
                  "epoch_seconds": time.perf_counter() - epoch_started,
                  "elapsed_seconds": time.perf_counter() - started}
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output / "best.pt")
        if args.model == "hrnet" and epoch in (10, 20):
            for group in optimizer.param_groups:
                group["lr"] *= .1

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    results = {"initial_val": initial_metrics, "best_epoch": best_epoch}
    for split, ids in (("train", train_idx), ("val", val_idx)):
        xy, confidence = predict(model, rgb, windows, ids, args.batch_size, device,
                                 args.model, grid_hw)
        split_rows = [rows[i] for i in ids]
        results[split] = evaluate_predictions(split_rows, xy, confidence)
        write_predictions(args.output / f"{split}_predictions.csv", split_rows, xy, confidence)
    results.update(elapsed_seconds=time.perf_counter() - started,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "val": validation_record(results["val"])}),
          flush=True)


if __name__ == "__main__":
    main()
