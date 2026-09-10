"""Cache dense local costs from an existing causal three-frame stage1 feature cache."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ballmotion.correspondence import cost_volume


VARIANTS = ("raw", "centered", "self_centered")
OFFSETS = {
    "delta1": [[dy, dx] for dy in range(-2, 3) for dx in range(-2, 3)],
    "delta2": [[dy, dx] for dy in range(-4, 5) for dx in range(-4, 5)],
}


def completed(path, source_cache, source_config, shape, variant):
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    expected = {"source_cache": str(source_cache), "source_config": source_config,
                "source_window_count": shape[0], "variant": variant,
                "shape": list(shape), "dtype": "float16"}
    mismatched = [key for key, value in expected.items() if meta.get(key) != value]
    if mismatched:
        raise ValueError(f"完整缓存条件不同，拒绝覆盖 {path}: {', '.join(mismatched)}")
    cost_path = path / "cost.npy"
    if not cost_path.exists():
        raise ValueError(f"完整缓存缺少 cost.npy: {path}")
    array = np.load(cost_path, mmap_mode="r")
    if array.shape != shape or array.dtype != np.float16:
        raise ValueError(f"完整缓存 shape/dtype 与元信息不符: {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "data/cache/tennis/"
                        "dinov3_convnext_tiny_512x288_step8_h2_s1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    source_cache = args.source_cache.resolve()
    meta_path, feature_path = source_cache / "metadata.json", source_cache / "stage1.npy"
    if not meta_path.exists() or not feature_path.exists():
        raise ValueError("Source metadata or stage1.npy is missing")
    source_meta = json.loads(meta_path.read_text())
    windows = np.asarray(source_meta["windows"], dtype=np.int64)
    source = np.load(feature_path, mmap_mode="r")
    if windows.ndim != 2 or windows.shape[1] != 3:
        raise ValueError("Source cache must contain Bx3 causal windows")
    if source.shape != tuple(source_meta["shapes"][1]) or source.shape[1:] != (192, 36, 64):
        raise ValueError("Source stage1 shape must match metadata and 192 x 36 x 64")
    if source.dtype != np.float16 or windows.min() < 0 or windows.max() >= len(source):
        raise ValueError("Source dtype or window indices are invalid")

    shape = (len(windows), 106, 36, 64)
    missing = [variant for variant in VARIANTS if not completed(
        args.output / variant, source_cache, source_meta["config"], shape, variant)]
    if not missing:
        print(f"使用已完成缓存: {args.output}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this cache job")

    arrays = {}
    for variant in missing:
        directory = args.output / variant
        directory.mkdir(parents=True, exist_ok=True)
        arrays[variant] = np.lib.format.open_memmap(
            directory / "cost.npy", mode="w+", dtype=np.float16, shape=shape)

    started = time.perf_counter()
    gpu_ms = {variant: 0.0 for variant in missing}
    conversion = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(windows), args.batch_size):
            stop = min(start + args.batch_size, len(windows))
            batch = torch.from_numpy(source[windows[start:stop]]).cuda().float()
            for variant in missing:
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record()
                costs = cost_volume(batch, variant)
                after.record()
                after.synchronize()
                gpu_ms[variant] += before.elapsed_time(after)
                stored = costs.half()
                if not torch.isfinite(costs).all() or not torch.isfinite(stored).all():
                    raise ValueError(f"Non-finite {variant} costs at target offset {start}")
                if start == 0:
                    conversion[variant] = {"first_batch_max_abs_error":
                                           float((costs - stored.float()).abs().max())}
                arrays[variant][start:stop] = stored.cpu().numpy()
                del costs, stored
            del batch
            if start % (50 * args.batch_size) == 0 or stop == len(windows):
                print(f"cached {stop}/{len(windows)} targets: {', '.join(missing)}", flush=True)

    for array in arrays.values():
        array.flush()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for variant in missing:
        meta = {
            "source_cache": str(source_cache),
            "source_config": source_meta["config"],
            "source_window_count": len(windows),
            "target_order": "source metadata windows order",
            "variant": variant,
            "shape": list(shape),
            "dtype": "float16",
            "batch_size": args.batch_size,
            "computed_variants": missing,
            "code_revision": revision,
            "offset_order": OFFSETS,
            "out_of_bounds_sentinel": -2,
            "fp16_conversion": conversion[variant],
            "gpu_compute_seconds": gpu_ms[variant] / 1000,
            "elapsed_seconds": elapsed,
            "timing_scope": ("elapsed includes source mmap reads, H2D, serial variant computation, "
                             "finite checks, fp16 D2H writes and flush; gpu_compute covers cost_volume only"),
            "peak_allocated_mib": peak,
            "device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
        }
        (args.output / variant / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"computed": missing, "shape": shape, "elapsed_seconds": elapsed,
                      "peak_allocated_mib": peak}), flush=True)


if __name__ == "__main__":
    main()
