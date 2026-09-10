"""按冻结特征缓存的唯一帧顺序保存确定性 RGB 输入，不展开重复窗口。"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cache_tennis_features import Images


def cache_rgb(source_cache, output, workers=2, batch_size=64):
    source_cache, output = Path(source_cache), Path(output)
    output_meta = output / "metadata.json"
    if output_meta.exists():
        metadata = json.loads(output_meta.read_text())
        if metadata["source_cache"] != str(source_cache.resolve()):
            raise ValueError("RGB 缓存来源不同；请用新目录保存")
        print(f"使用已完成 RGB 缓存: {output} ({len(metadata['frames'])} frames)", flush=True)
        return metadata

    source = json.loads((source_cache / "metadata.json").read_text())
    config, frames, windows = source["config"], source["frames"], source["windows"]
    height, width = config["input_hw"]
    output.mkdir(parents=True, exist_ok=True)
    rgb = np.lib.format.open_memmap(output / "rgb.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(frames), 3, height, width))
    loader = DataLoader(Images(Path(config["data_root"]), frames, (height, width)),
                        batch_size=batch_size, shuffle=False, num_workers=workers)
    started, written = time.perf_counter(), 0
    for batch_id, pixels in enumerate(loader):
        rgb[written:written + len(pixels)] = pixels.numpy()
        written += len(pixels)
        if batch_id % 20 == 0 or written == len(frames):
            print(f"cached {written}/{len(frames)} unique RGB frames, "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    rgb.flush()
    metadata = {
        "source_cache": str(source_cache.resolve()),
        "source_config": config,
        "frames": frames,
        "windows": windows,
        "shape": list(rgb.shape),
        "input_hw": [height, width],
        "dtype": "uint8",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = cache_rgb(args.source_cache, args.output)
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("frames", "windows")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
