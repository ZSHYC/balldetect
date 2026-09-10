"""验证缓存量化是否改变已训练探针的球定位，而非只看全图特征误差。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cache_tennis_features import ROOT, Images, ConvNeXt
from ballmotion.probe import SpatialProbe, evaluate_predictions, frame_batch
from ballmotion.tennis import grid_to_original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="同一缓存上的已完成读出运行")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    meta = json.loads((args.cache / "metadata.json").read_text())
    all_rows = meta["frames"]
    windows = np.asarray(meta["windows"], dtype=np.int64) if "windows" in meta else np.arange(len(all_rows))[:, None]
    windows = windows[[all_rows[w[-1]]["game"] == "game7" for w in windows]]
    rows = [all_rows[w[-1]] for w in windows]
    frame_ids, compact = np.unique(windows, return_inverse=True)
    compact = compact.reshape(windows.shape)
    models, configs = [], []
    for run in args.runs:
        config = json.loads((run / "config.json").read_text())
        stage = config["stage"]
        if config["cache_config"] != meta["config"]:
            raise ValueError("运行缓存条件不匹配")
        num_frames = config.get("num_frames", 1)
        model = SpatialProbe(meta["shapes"][stage][1] * num_frames, upscale=config.get("upscale", 1),
                             hidden_channels=config.get("hidden_channels", 0), num_frames=num_frames).cuda().eval()
        model.load_state_dict(torch.load(run / "best.pt", weights_only=True, map_location="cuda")["model"])
        models.append(model)
        configs.append(config)
    backbone = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    backbone.load_state_dict(torch.load(ROOT / meta["config"]["weights"], weights_only=True, mmap=True), strict=True)
    backbone.requires_grad_(False).cuda().eval()
    loader = DataLoader(Images(Path(meta["config"]["data_root"]), [all_rows[i] for i in frame_ids],
                               tuple(meta["config"]["input_hw"])),
                        batch_size=8, num_workers=2, shuffle=False)
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    positions = [[[], []] for _ in models]
    probabilities = [[[], []] for _ in models]
    # ponytail: 开发验证帧驻留CPU内存；验证集扩大到内存放不下时再按clip处理。
    stages = sorted({c["stage"] for c in configs})
    full_precision = {s: np.empty((len(frame_ids), *meta["shapes"][s][1:]), dtype=np.float32) for s in stages}
    cached = {s: np.load(args.cache / f"stage{s}.npy", mmap_mode="r") for s in stages}
    offset = 0
    with torch.inference_mode():
        for pixels in loader:
            features = backbone.get_intermediate_layers((pixels.cuda().float() / 255 - mean) / std,
                                                        n=4, reshape=True, norm=False)
            for stage in stages:
                full_precision[stage][offset:offset + len(pixels)] = features[stage].cpu().numpy()
            offset += len(pixels)
        for index, (config, model) in enumerate(zip(configs, models)):
            stage = config["stage"]
            mode = config.get("temporal_input", "current")
            for start in range(0, len(rows), 16):
                inputs = (frame_batch(full_precision[stage], compact[start:start + 16], mode),
                          frame_batch(cached[stage], windows[start:start + 16], mode))
                for precision, x in enumerate(inputs):
                    x = torch.from_numpy(x).cuda().float()
                    logits = model(x)
                    positions[index][precision].extend(logits[:, :-1].argmax(1).cpu().tolist())
                    probabilities[index][precision].extend((1 - logits.softmax(1)[:, -1]).cpu().tolist())
    results = {"frames": len(rows), "unique_input_frames": len(frame_ids),
               "scope": "game7 target windows; fixed trained heads; recomputed float32 vs actual float16 cache, including extraction differences",
               "by_run": {}}
    for index, (run, config) in enumerate(zip(args.runs, configs)):
        stage = config["stage"]
        p32, p16 = np.array(positions[index])
        q32, q16 = np.array(probabilities[index])
        grid_hw = tuple(s * models[index].upscale for s in meta["shapes"][stage][-2:])
        results["by_run"][str(run)] = {
            "stage": stage, "temporal_input": config.get("temporal_input", "current"),
            "changed_spatial_argmax": int(np.sum(p32 != p16)),
            "changed_presence_at_half": int(np.sum((q32 >= .5) != (q16 >= .5))),
            "max_presence_probability_difference": float(np.max(np.abs(q32 - q16))),
            "float32": evaluate_predictions(rows, grid_to_original(p32, grid_hw), q32)["location"],
            "float16": evaluate_predictions(rows, grid_to_original(p16, grid_hw), q16)["location"]}
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
