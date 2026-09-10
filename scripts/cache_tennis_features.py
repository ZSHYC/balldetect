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
from ballmotion.tennis import read_frames
from dinov3.models.convnext import ConvNeXt


class Images(Dataset):
    def __init__(self, root, rows, hw):
        self.root, self.rows, self.hw = root, rows, hw

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]["image_path"]) as im:
            im = im.convert("RGB").resize(self.hw[::-1], Image.Resampling.BILINEAR)
            return torch.from_numpy(np.array(im)).permute(2, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/tracknet_tennis")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frame-step", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0, help="仅供 smoke；0 为完整开发采样")
    args = parser.parse_args()
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
    meta_path = args.output / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["config"] != config:
            raise ValueError("缓存配置不同；请用新目录保存新条件")
        print(f"使用已完成缓存: {args.output} ({len(meta['frames'])} frames)", flush=True)
        return
    rows, excluded = read_frames(args.data_root, set(range(1, 8)), args.frame_step)
    if args.max_frames:
        # smoke 仍包含两个 split，不能截到只有训练比赛。
        train = [r for r in rows if r["game"] != "game7"][:args.max_frames // 2]
        val = [r for r in rows if r["game"] == "game7"][:args.max_frames - len(train)]
        rows = train + val
    if not rows:
        raise ValueError("没有选中的帧")
    if any((r["height"], r["width"]) != (720, 1280) for r in rows):
        raise ValueError("此协议仅支持发布的 1280x720 图像")
    args.output.mkdir(parents=True, exist_ok=True)
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True, mmap=True), strict=True)
    model.requires_grad_(False).eval().cuda()
    loader = DataLoader(Images(args.data_root, rows, (args.height, args.width)),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    arrays, conversion = [], []
    position, forward_ms = 0, 0.
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_id, pixels in enumerate(loader):
            x = (pixels.cuda(non_blocking=True).float() / 255 - mean) / std
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before.record()
            features = model.get_intermediate_layers(x, n=4, reshape=True, norm=False)
            after.record()
            after.synchronize()
            forward_ms += before.elapsed_time(after)
            if not arrays:
                for stage, f in enumerate(features):
                    saved = f.half().float()
                    relative = float(torch.linalg.vector_norm(f - saved) / torch.linalg.vector_norm(f))
                    if not torch.isfinite(saved).all() or relative > .001:
                        raise ValueError(f"stage {stage}: float16 保存误差过大，应改用 float32")
                    conversion.append(dict(relative_l2=relative, max_abs_error=float((f - saved).abs().max())))
                    shape = (len(rows), *f.shape[1:])
                    arrays.append(np.lib.format.open_memmap(args.output / f"stage{stage}.npy", mode="w+",
                                                           dtype=np.float16, shape=shape))
                print(json.dumps({"shapes": [list(a.shape) for a in arrays],
                                  "storage_bytes": sum(a.nbytes for a in arrays),
                                  "fp16_conversion": conversion}), flush=True)
            for array, f in zip(arrays, features):
                values = f.half().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite features at frame offset {position}")
                array[position:position + len(pixels)] = values
            position += len(pixels)
            if batch_id % 20 == 0 or position == len(rows):
                print(f"cached {position}/{len(rows)} frames, elapsed={time.perf_counter()-started:.1f}s", flush=True)
    for array in arrays:
        array.flush()
    meta = dict(config=config, frames=rows, excluded=excluded, shapes=[list(a.shape) for a in arrays],
                by_game=dict(Counter(r["game"] for r in rows)), fp16_conversion=conversion,
                device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                elapsed_seconds=time.perf_counter() - started, forward_seconds=forward_ms / 1000,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("frames", "excluded")}), flush=True)


if __name__ == "__main__":
    main()
