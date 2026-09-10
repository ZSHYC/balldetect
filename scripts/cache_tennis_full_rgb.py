"""构造 games1-7 全量因果三帧 RGB 缓存，复用已有同条件像素。"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ballmotion.tennis import read_frames
from cache_tennis_features import Images, frame_key, select_cache_rows


INPUT_HW = (288, 512)


def cache_full_rgb(data_root, output, reuse_cache, workers=2, batch_size=64):
    data_root, output, reuse_cache = Path(data_root), Path(output), Path(reuse_cache)
    config = {"data_root": str(data_root.resolve()), "input_hw": list(INPUT_HW),
              "resize": "PIL bilinear RGB", "games": list(range(1, 8)),
              "target_step": 1, "history": 2}
    meta_path = output / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["config"] != config:
            raise ValueError("已有 RGB 缓存配置不同；请使用新目录")
        print(f"使用已完成全量 RGB 缓存: {output} ({len(metadata['frames'])} frames)", flush=True)
        return metadata

    valid, excluded = read_frames(data_root, set(range(1, 8)), 1)
    full_rows = sorted(valid + excluded,
                       key=lambda r: (int(r["game"][4:]), int(r["clip"][4:]),
                                      int(r["original_frame_id"])))
    frames, windows, boundary_excluded = select_cache_rows(full_rows, 1, 2, 0)
    if any((r["height"], r["width"]) != (720, 1280) for r in frames):
        raise ValueError("全量 Tennis 协议仅支持发布的 1280x720 图像")

    reuse_meta = json.loads((reuse_cache / "metadata.json").read_text())
    reuse_config = reuse_meta["source_config"]
    for key, expected in (("data_root", config["data_root"]),
                          ("input_hw", config["input_hw"]),
                          ("resize", config["resize"])):
        if reuse_config[key] != expected:
            raise ValueError(f"复用 RGB 缓存的 {key} 条件不同")
    reuse_rgb = np.load(reuse_cache / "rgb.npy", mmap_mode="r")
    if (list(reuse_rgb.shape) != reuse_meta["shape"] or reuse_rgb.dtype != np.uint8
            or tuple(reuse_rgb.shape[1:]) != (3, *INPUT_HW)):
        raise ValueError("复用 RGB 数组 shape/dtype 与元信息不符")
    lookup = {}
    for index, row in enumerate(reuse_meta["frames"]):
        key = frame_key(row)
        if key in lookup:
            raise ValueError(f"复用 RGB 缓存包含重复真实帧: {key}")
        lookup[key] = index
    reuse_pairs = [(index, lookup[frame_key(row)]) for index, row in enumerate(frames)
                   if frame_key(row) in lookup]
    reused_positions = {index for index, _ in reuse_pairs}
    missing_positions = [i for i in range(len(frames)) if i not in reused_positions]

    output.mkdir(parents=True, exist_ok=True)
    rgb = np.lib.format.open_memmap(output / "rgb.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(frames), 3, *INPUT_HW))
    started = time.perf_counter()
    for start in range(0, len(reuse_pairs), 64):
        batch = reuse_pairs[start:start + 64]
        output_ids, source_ids = zip(*batch)
        rgb[list(output_ids)] = reuse_rgb[list(source_ids)]
    if reuse_pairs:
        print(f"reused {len(reuse_pairs)}/{len(frames)} unique RGB frames", flush=True)

    loader = DataLoader(Images(data_root, [frames[i] for i in missing_positions], INPUT_HW),
                        batch_size=batch_size, shuffle=False, num_workers=workers)
    decoded = 0
    for batch_id, pixels in enumerate(loader):
        positions = missing_positions[decoded:decoded + len(pixels)]
        rgb[positions] = pixels.numpy()
        decoded += len(pixels)
        if batch_id % 20 == 0 or decoded == len(missing_positions):
            print(f"decoded {decoded}/{len(missing_positions)} missing frames, "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    rgb.flush()
    target_rows = [frames[window[-1]] for window in windows]
    metadata = {
        "config": config, "frames": frames, "windows": windows.tolist(),
        "shape": list(rgb.shape), "input_hw": list(INPUT_HW), "dtype": "uint8",
        "target_count": len(windows),
        "train_targets": sum(row["game"] != "game7" for row in target_rows),
        "val_targets": sum(row["game"] == "game7" for row in target_rows),
        "excluded_label_rows": len(excluded),
        "boundary_excluded_targets": boundary_excluded,
        "reused_frames": len(reuse_pairs), "decoded_frames": len(missing_positions),
        "reuse_cache": str(reuse_cache.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "reuse copy + missing-frame PIL decode/resize + RGB flush; excludes frames.csv selection and metadata write",
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("frames", "windows")},
                     ensure_ascii=False), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/tracknet_tennis")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-cache", type=Path, required=True)
    args = parser.parse_args()
    cache_full_rgb(args.data_root, args.output, args.reuse_cache)


if __name__ == "__main__":
    main()
