"""验证缓存量化是否改变已训练探针的球定位，而非只看全图特征误差。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from cache_tennis_features import ROOT, Images, ConvNeXt
from ballmotion.probe import SpatialProbe, evaluate_predictions
from ballmotion.tennis import grid_to_original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="已完成的不同 stage 运行")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    meta = json.loads((args.cache / "metadata.json").read_text())
    rows = [r for r in meta["frames"] if r["game"] == "game7"]
    models, stages = [], []
    for run in args.runs:
        config = json.loads((run / "config.json").read_text())
        stage = config["stage"]
        if stage in stages or config["cache_config"] != meta["config"]:
            raise ValueError("运行层位/缓存条件不匹配")
        model = SpatialProbe(meta["shapes"][stage][1], upscale=config.get("upscale", 1),
                             hidden_channels=config.get("hidden_channels", 0)).cuda().eval()
        model.load_state_dict(torch.load(run / "best.pt", weights_only=True, map_location="cuda")["model"])
        models.append(model)
        stages.append(stage)
    backbone = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    backbone.load_state_dict(torch.load(ROOT / meta["config"]["weights"], weights_only=True, mmap=True), strict=True)
    backbone.requires_grad_(False).cuda().eval()
    loader = DataLoader(Images(Path(meta["config"]["data_root"]), rows, tuple(meta["config"]["input_hw"])),
                        batch_size=8, num_workers=2, shuffle=False)
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    positions = [[[], []] for _ in models]
    probabilities = [[[], []] for _ in models]
    with torch.inference_mode():
        for pixels in loader:
            features = backbone.get_intermediate_layers((pixels.cuda().float() / 255 - mean) / std,
                                                        n=4, reshape=True, norm=False)
            for index, (stage, model) in enumerate(zip(stages, models)):
                feature = features[stage]
                for precision, x in enumerate((feature, feature.half().float())):
                    logits = model(x)
                    positions[index][precision].extend(logits[:, :-1].argmax(1).cpu().tolist())
                    probabilities[index][precision].extend((1 - logits.softmax(1)[:, -1]).cpu().tolist())
    results = {"frames": len(rows), "scope": "game7 sampled validation; fixed trained heads; float32 forward vs float16 storage",
               "runs": [str(p) for p in args.runs], "stages": {}}
    for index, stage in enumerate(stages):
        p32, p16 = np.array(positions[index])
        q32, q16 = np.array(probabilities[index])
        grid_hw = tuple(s * models[index].upscale for s in meta["shapes"][stage][-2:])
        results["stages"][str(stage)] = {
            "changed_spatial_argmax": int(np.sum(p32 != p16)),
            "changed_presence_at_half": int(np.sum((q32 >= .5) != (q16 >= .5))),
            "max_presence_probability_difference": float(np.max(np.abs(q32 - q16))),
            "float32": evaluate_predictions(rows, grid_to_original(p32, grid_hw), q32)["location"],
            "float16": evaluate_predictions(rows, grid_to_original(p16, grid_hw), q16)["location"]}
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
