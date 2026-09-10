"""一次解码/前向，保存开发集的确定性冻结特征。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
from ballmotion.tennis import causal_windows, read_frames
from dinov3.models.convnext import ConvNeXt


FEATURE_CONFIG_KEYS = ("backbone", "weights", "upstream_revision", "input_hw", "data_root", "resize",
                       "normalize", "stage_norm", "forward_dtype", "storage_dtype")


class Images(Dataset):
    def __init__(self, root, rows, hw):
        self.root, self.rows, self.hw = root, rows, hw

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]["image_path"]) as im:
            im = im.convert("RGB").resize(self.hw[::-1], Image.Resampling.BILINEAR)
            return torch.from_numpy(np.array(im)).permute(2, 0, 1)


def frame_key(row):
    return row["game"], row["clip"], int(row["original_frame_id"]), row["image_path"]


def select_cache_rows(rows, target_step, history, max_frames):
    windows, boundary_excluded = causal_windows(rows, target_step, history)
    if max_frames:
        train = [w for w in windows if rows[w[-1]]["game"] != "game7"][:max_frames // 2]
        val = [w for w in windows if rows[w[-1]]["game"] == "game7"][:max_frames - len(train)]
        windows = np.asarray(train + val, dtype=np.int64).reshape(-1, history + 1)
    if not len(windows):
        raise ValueError("没有选中的目标窗口")
    source_indices = sorted(set(windows.reshape(-1).tolist()))
    remap = {source: target for target, source in enumerate(source_indices)}
    compact_windows = np.asarray([[remap[i] for i in window] for window in windows], dtype=np.int64)
    return [rows[i] for i in source_indices], compact_windows, boundary_excluded


def open_reuse_source(path, config, selected_stages):
    if path is None:
        return None, {}, {}
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"复用缓存未完成或不存在: {path}")
    meta = json.loads(meta_path.read_text())
    mismatched = [key for key in FEATURE_CONFIG_KEYS if meta.get("config", {}).get(key) != config[key]]
    if mismatched:
        raise ValueError(f"复用缓存的特征条件不同: {', '.join(mismatched)}")
    arrays = {}
    for stage in selected_stages:
        feature_path = path / f"stage{stage}.npy"
        if not feature_path.exists():
            return meta, {}, {}
        array = np.load(feature_path, mmap_mode="r")
        expected = tuple(meta["shapes"][stage])
        if array.shape != expected or array.dtype != np.dtype(config["storage_dtype"]):
            raise ValueError(f"复用缓存 stage{stage} 的 shape/dtype 与元信息不符")
        arrays[stage] = array
    lookup = {}
    for i, row in enumerate(meta["frames"]):
        key = frame_key(row)
        if key in lookup:
            raise ValueError(f"复用缓存包含重复真实帧: {key}")
        lookup[key] = i
    return meta, arrays, lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/tracknet_tennis")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frame-step", type=int, default=8)
    parser.add_argument("--stages", type=int, nargs="+", choices=range(4), help="只保存实际使用的层，默认四层")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0, help="仅供 smoke；0 为完整开发采样")
    parser.add_argument("--history-frames", type=int, default=0, help="目标帧之前的真实连续帧数")
    parser.add_argument("--reuse-cache", type=Path, help="复用相同视觉条件下已缓存的真实帧")
    args = parser.parse_args()
    selected_stages = args.stages if args.stages is not None else list(range(4))
    if args.history_frames < 0:
        raise ValueError("history-frames must be nonnegative")
    if args.height % 32 or args.width % 32 or args.width / args.height != 1280 / 720:
        raise ValueError("需要保持 16:9，且 H/W 为 32 的倍数")
    torch.set_num_threads(4)
    weights = ROOT / "models/pretrained/dinov3/lvd1689m/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    upstream = subprocess.check_output(["git", "-C", str(ROOT / "third_party/dinov3"),
                                        "rev-parse", "HEAD"], text=True).strip()
    config = dict(backbone="dinov3_convnext_tiny", weights=str(weights.relative_to(ROOT)),
                  upstream_revision=upstream, input_hw=[args.height, args.width],
                  frame_step=args.frame_step, games=list(range(1, 8)), max_frames=args.max_frames,
                  data_root=str(args.data_root.resolve()), resize="PIL bilinear RGB",
                  normalize="ImageNet", stage_norm=False, forward_dtype="float32", storage_dtype="float16")
    if args.stages is not None:
        config["saved_stages"] = args.stages
    if args.history_frames:
        config["history_frames"] = args.history_frames
    meta_path = args.output / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["config"] != config:
            raise ValueError("缓存配置不同；请用新目录保存新条件")
        print(f"使用已完成缓存: {args.output} ({len(meta['frames'])} frames)", flush=True)
        return
    valid, excluded = read_frames(args.data_root, set(range(1, 8)), 1)
    full_rows = sorted(valid + excluded,
                       key=lambda r: (int(r["game"][4:]), int(r["clip"][4:]), int(r["original_frame_id"])))
    rows, windows, boundary_excluded = select_cache_rows(
        full_rows, args.frame_step, args.history_frames, args.max_frames)
    if any((r["height"], r["width"]) != (720, 1280) for r in rows):
        raise ValueError("此协议仅支持发布的 1280x720 图像")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reuse_meta, reuse_arrays, reuse_lookup = open_reuse_source(args.reuse_cache, config, selected_stages)
    reuse_pairs = [(i, reuse_lookup[frame_key(row)]) for i, row in enumerate(rows)
                   if frame_key(row) in reuse_lookup]
    reused_positions = {i for i, _ in reuse_pairs}
    missing_positions = [i for i in range(len(rows)) if i not in reused_positions]
    arrays, conversion, shapes = {}, [], None
    if reuse_pairs:
        shapes = [(len(rows), *shape[1:]) for shape in reuse_meta["shapes"]]
        conversion = reuse_meta["fp16_conversion"]
        for stage in selected_stages:
            arrays[stage] = np.lib.format.open_memmap(
                args.output / f"stage{stage}.npy", mode="w+", dtype=np.float16, shape=shapes[stage])
        for start in range(0, len(reuse_pairs), 64):
            batch = reuse_pairs[start:start + 64]
            output_ids, source_ids = zip(*batch)
            for stage, array in arrays.items():
                array[list(output_ids)] = reuse_arrays[stage][list(source_ids)]
    forward_ms, peak_allocated = 0., 0.
    if missing_positions:
        model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True, mmap=True), strict=True)
        model.requires_grad_(False).eval().cuda()
        loader = DataLoader(Images(args.data_root, [rows[i] for i in missing_positions], (args.height, args.width)),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
        torch.cuda.reset_peak_memory_stats()
        computed = 0
        with torch.inference_mode():
            for batch_id, pixels in enumerate(loader):
                x = (pixels.cuda(non_blocking=True).float() / 255 - mean) / std
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record()
                features = model.get_intermediate_layers(x, n=4, reshape=True, norm=False)
                after.record()
                after.synchronize()
                forward_ms += before.elapsed_time(after)
                if shapes is None:
                    shapes = [(len(rows), *f.shape[1:]) for f in features]
                    for stage, f in enumerate(features):
                        saved = f.half().float()
                        relative = float(torch.linalg.vector_norm(f - saved) / torch.linalg.vector_norm(f))
                        if not torch.isfinite(saved).all() or relative > .001:
                            raise ValueError(f"stage {stage}: float16 保存误差过大，应改用 float32")
                        conversion.append(dict(relative_l2=relative,
                                               max_abs_error=float((f - saved).abs().max())))
                        if stage in selected_stages:
                            arrays[stage] = np.lib.format.open_memmap(
                                args.output / f"stage{stage}.npy", mode="w+", dtype=np.float16,
                                shape=shapes[stage])
                    print(json.dumps({"shapes": shapes, "saved_stages": selected_stages,
                                      "storage_bytes": sum(a.nbytes for a in arrays.values()),
                                      "fp16_conversion": conversion}), flush=True)
                batch_positions = missing_positions[computed:computed + len(pixels)]
                for stage, array in arrays.items():
                    values = features[stage].half().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite features at frame offset {computed}")
                    array[batch_positions] = values
                computed += len(pixels)
                if batch_id % 20 == 0 or computed == len(missing_positions):
                    print(f"computed {computed}/{len(missing_positions)} missing frames, "
                          f"reused={len(reuse_pairs)}, elapsed={time.perf_counter()-started:.1f}s", flush=True)
        peak_allocated = torch.cuda.max_memory_allocated() / 2**20
    if shapes is None:
        raise ValueError("既没有可复用特征，也没有待计算帧")
    for array in arrays.values():
        array.flush()
    meta = dict(config=config, frames=rows, windows=windows.tolist(), excluded=excluded, shapes=shapes,
                by_game=dict(Counter(r["game"] for r in rows)), fp16_conversion=conversion,
                target_count=len(windows), boundary_excluded_targets=boundary_excluded,
                reused_frames=len(reuse_pairs), computed_frames=len(missing_positions),
                device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                elapsed_seconds=time.perf_counter() - started, forward_seconds=forward_ms / 1000,
                timing_scope="reuse/open/copy + model setup + missing-frame extraction + flush; excludes index/metadata I/O",
                peak_allocated_mib=peak_allocated)
    if args.reuse_cache is not None:
        meta["reuse_cache"] = str(args.reuse_cache.resolve())
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("frames", "windows", "excluded")}), flush=True)


if __name__ == "__main__":
    main()
