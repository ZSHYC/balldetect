"""冻结完整DINO基线，仅训练当前stage0的native/pooled位置残差。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.probe import evaluate_predictions
from ballmotion.tennis import grid_targets, grid_to_original
from compare_predictions import read_predictions
from train_tennis_heatmap import build_dino_model, validation_record, write_predictions


class DetailResidual(nn.Module):
    def __init__(self, detail):
        super().__init__()
        self.detail = detail
        self.norm = nn.GroupNorm(1, 96, affine=False)
        self.location = nn.Sequential(nn.Conv2d(96, 32, 1), nn.GELU(),
                                      nn.Conv2d(32, 1, 3, padding=1, bias=False))
        nn.init.zeros_(self.location[2].weight)

    def forward(self, features, logits):
        features = self.norm(features)
        if self.detail == "pooled":
            features = F.interpolate(F.avg_pool2d(features, 2), size=features.shape[-2:],
                                     mode="bilinear", align_corners=False)
        residual = self.location(features).flatten(1)
        return torch.cat((logits[:, :-1] + residual, logits[:, -1:]), dim=1)


@torch.no_grad()
def current_features(base, rgb, windows, ids, device):
    pixels = torch.from_numpy(rgb[windows[ids, -1]]).to(device)
    normalized = (pixels.float() / 255 - base.mean) / base.std
    return base.prefix[1](base.prefix[0](normalized))


def decode_logits(logits):
    xy = grid_to_original(logits[:, :-1].argmax(1).cpu().numpy(), (72, 128))
    probability = (1 - logits.softmax(1)[:, -1]).cpu().numpy()
    return xy, probability


def identities(rows):
    return [[r["game"], r["clip"], int(r["original_frame_id"])] for r in rows]


def fixed_logits(base, rgb, windows, rows, splits, cache, base_run, source, epoch, device):
    expected = {"base_run": str(base_run.resolve()), "source_config": source,
                "checkpoint_epoch": epoch, "target_identities": identities(rows),
                "shape": [len(rows), 72 * 128 + 1], "dtype": "float32"}
    metadata_path = cache / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("固定logits来源或目标顺序与本次实验不符")
        logits = np.load(cache / "logits.npy", mmap_mode="r")
        if list(logits.shape) != expected["shape"] or logits.dtype != np.float32:
            raise ValueError("固定logits的shape/dtype不符")
        print(json.dumps({"fixed_logits": "reuse", "path": str(cache)}), flush=True)
        return logits, metadata
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "logits.npy").exists():
        raise ValueError("存在未完成的logits缓存；先处理该次中断产物")
    logits = np.lib.format.open_memmap(cache / "logits.npy", mode="w+", dtype=np.float32,
                                       shape=tuple(expected["shape"]))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for split, indices in splits.items():
            for start in range(0, len(indices), 8):
                ids = indices[start:start + 8]
                unique, inverse = np.unique(windows[ids], return_inverse=True)
                features = base.encode(torch.from_numpy(rgb[unique]).to(device))
                restored = features[torch.from_numpy(inverse).to(device)].reshape(
                    len(ids), 576, 36, 64)
                logits[ids] = base.head(restored).cpu().numpy()
                if start % 1600 == 0 or start + len(ids) == len(indices):
                    print(json.dumps({"cache_split": split, "processed": start + len(ids),
                                      "total": len(indices)}), flush=True)
    logits.flush()
    replay = {}
    for split, indices in splits.items():
        saved_rows, saved_xy, saved_q = read_predictions(base_run / f"{split}_predictions.csv")
        if identities(saved_rows) != identities([rows[i] for i in indices]):
            raise ValueError(f"{split}的基线预测身份与缓存顺序不同")
        xy, q = [], []
        with torch.inference_mode():
            for start in range(0, len(indices), 8):
                ids = indices[start:start + 8]
                batch_xy, batch_q = decode_logits(torch.from_numpy(logits[ids]).to(device))
                xy.append(batch_xy)
                q.append(batch_q)
        xy, q = np.concatenate(xy), np.concatenate(q)
        if (not np.array_equal(xy, saved_xy)
                or not np.allclose(q, saved_q, atol=1e-6, rtol=0)
                or not np.array_equal(q >= .5, saved_q >= .5)):
            raise ValueError(f"{split}的固定logits不能复现原位置/presence")
        replay[split] = {"targets": len(indices), "changed_locations": 0,
                         "max_probability_difference": float(np.max(np.abs(q - saved_q)))}
    metadata = {**expected, "replay": replay,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "array_bytes": int(logits.nbytes),
                "code_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "timing_scope": "original batch8 train/val prediction, float32 write and prediction replay; excludes model/RGB metadata loading"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("replay", "elapsed_seconds",
                                             "peak_allocated_mib", "array_bytes")}), flush=True)
    del logits
    return np.load(cache / "logits.npy", mmap_mode="r"), metadata


@torch.no_grad()
def predict_detail(base, detail, rgb, windows, cached, indices, batch_size, device):
    detail.eval()
    xy, probability = [], []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        features = current_features(base, rgb, windows, ids, device)
        logits = detail(features, torch.from_numpy(cached[ids]).to(device))
        batch_xy, batch_probability = decode_logits(logits)
        xy.append(batch_xy)
        probability.append(batch_probability)
    return np.concatenate(xy), np.concatenate(probability)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, default=Path("outputs/full_heatmap/dino_prefix_seed0"))
    parser.add_argument("--logit-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detail", choices=("native", "pooled"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.output / "config.json").exists():
        raise ValueError("此目录已有实验配置；使用新目录避免覆盖")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    source = json.loads((args.base_run / "config.json").read_text())
    source_results = json.loads((args.base_run / "results.json").read_text())
    if (source["model"] != "dino" or source.get("auxiliary", "none") != "none"
            or source.get("temporal_input", "history") != "history"
            or source_results["best_epoch"] != 7):
        raise ValueError("本协议要求完整无辅助history的epoch7基线")
    rgb_cache = ROOT / source["rgb_cache"]
    metadata = json.loads((rgb_cache / "metadata.json").read_text())
    if metadata["config"] != source["cache_config"]:
        raise ValueError("RGB缓存配置与固定基线不同")
    frames = metadata["frames"]
    if set(r["game"] for r in frames) != {f"game{i}" for i in range(1, 8)}:
        raise ValueError("只允许games1–7开发数据")
    windows = np.asarray(metadata["windows"], dtype=np.int64)
    if windows.ndim != 2 or windows.shape[1] != 3:
        raise ValueError("需要真实三帧窗口")
    rows = [frames[i] for i in windows[:, -1]]
    splits = {split: np.array([i for i, r in enumerate(rows)
                                if (r["game"] == "game7") == (split == "val")])
              for split in ("train", "val")}
    if (len(splits["train"]), len(splits["val"])) != (12167, 1863):
        raise ValueError("需要完整12,167/1,863目标")
    rgb = np.load(rgb_cache / "rgb.npy", mmap_mode="r")
    if rgb.shape != (len(frames), 3, 288, 512) or rgb.dtype != np.uint8:
        raise ValueError("需要原512×288 uint8 RGB缓存")
    base = build_dino_model(source["weights"])
    checkpoint = torch.load(args.base_run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != 7:
        raise ValueError("基线checkpoint与结果epoch不符")
    base.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    base.requires_grad_(False).eval().to(device)
    cached, cache_metadata = fixed_logits(base, rgb, windows, rows, splits, args.logit_cache,
                                          args.base_run, source, 7, device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    detail = DetailResidual(args.detail).to(device)
    optimizer = torch.optim.AdamW(detail.parameters(), lr=3e-4, weight_decay=.01)
    target_xy = np.asarray([[r["x_raw"], r["y_raw"]] for r in rows], dtype=float)
    targets = grid_targets(target_xy, np.asarray([r["visibility_raw"] != 0 for r in rows]),
                           (72, 128))
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                         text=True).strip(),
              "rgb_cache": source["rgb_cache"], "base_code_revision": source["code_revision"],
              "base_epoch": 7, "train_frames": len(splits["train"]), "val_frames": len(splits["val"]),
              "input_slots": ["t-2", "t-1", "t"], "detail_slot": "t", "target_slot": 2,
              "model_name": "frozen DINO stage0 current-frame residual readout",
              "detail_config": {"normalization": "GroupNorm(1,96,affine=False) before pooling",
                                "projection": "96->32 1x1 bias, GELU, 32->1 3x3 without bias",
                                "initialization": "last convolution zero; other layer torch seed",
                                "pooled": "avg_pool2d(2,2), bilinear align_corners=False to 72x128"},
              "trainable_parameters": sum(p.numel() for p in detail.parameters()),
              "total_parameters": sum(p.numel() for p in base.parameters())
                                  + sum(p.numel() for p in detail.parameters()),
              "frozen": "whole original prefix and SpatialProbe; eval mode",
              "loss": "cross_entropy over 9216 spatial + 1 absence classes",
              "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": .01, "schedule": "constant"},
              "selection": source["selection"], "precision": "float32; no AMP", "augmentation": None,
              "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
              "fixed_logits_preparation_seconds": cache_metadata["elapsed_seconds"],
              "timing_scope": "RGB and fixed logits mmap indexing/H2D, frozen current stage0, residual train/evaluation; excludes fixed logits preparation; not deployment timing"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(config, ensure_ascii=False), flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    val_rows = [rows[i] for i in splits["val"]]
    xy, q = predict_detail(base, detail, rgb, windows, cached, splits["val"], args.batch_size, device)
    initial = evaluate_predictions(val_rows, xy, q)
    if initial != source_results["val"]:
        raise ValueError("零残差epoch0未复现固定基线的验证结果")
    best = (initial["detection16"]["f1"], initial["detection8"]["f1"])
    best_epoch = 0
    torch.save({"residual": detail.state_dict(), "epoch": 0}, args.output / "best.pt")

    def record_epoch(epoch, loss, metrics, epoch_started):
        record = {"epoch": epoch, "train_loss": loss, "val": validation_record(metrics),
                  "epoch_seconds": time.perf_counter() - epoch_started,
                  "elapsed_seconds": time.perf_counter() - started}
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    record_epoch(0, None, initial, started)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        detail.train()
        order = rng.permutation(splits["train"])
        loss_sum = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            features = current_features(base, rgb, windows, ids, device)
            logits = detail(features, torch.from_numpy(cached[ids]).to(device))
            loss = F.cross_entropy(logits, torch.from_numpy(targets[ids]).to(device))
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            processed = start + len(ids)
            if (start // args.batch_size + 1) % 400 == 0 or processed == len(order):
                print(json.dumps({"epoch": epoch, "processed": processed, "total": len(order),
                                  "train_loss_so_far": loss_sum / processed}), flush=True)
        xy, q = predict_detail(base, detail, rgb, windows, cached, splits["val"], args.batch_size, device)
        metrics = evaluate_predictions(val_rows, xy, q)
        record_epoch(epoch, loss_sum / len(order), metrics, epoch_started)
        score = (metrics["detection16"]["f1"], metrics["detection8"]["f1"])
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"residual": detail.state_dict(), "epoch": epoch}, args.output / "best.pt")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    detail.load_state_dict(checkpoint["residual"])
    results = {"initial_val": initial, "best_epoch": best_epoch}
    for split, indices in splits.items():
        xy, q = predict_detail(base, detail, rgb, windows, cached, indices, args.batch_size, device)
        split_rows = [rows[i] for i in indices]
        results[split] = evaluate_predictions(split_rows, xy, q)
        write_predictions(args.output / f"{split}_predictions.csv", split_rows, xy, q)
    results.update(elapsed_seconds=time.perf_counter() - started,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "val": validation_record(results["val"])}), flush=True)


if __name__ == "__main__":
    main()
