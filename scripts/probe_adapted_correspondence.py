"""用已选中适配前缀复算固定 GT-query 对应诊断。"""
import argparse
from collections import defaultdict
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
from dinov3.models.convnext import ConvNeXt
from probe_tennis_correspondence import analyze_pair, summarize, summarize_groups


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_prefix(run):
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    prefix_state = {key.removeprefix("prefix."): value
                    for key, value in state.items() if key.startswith("prefix.")}
    if not prefix_state or "mean" not in state or "std" not in state:
        raise ValueError("checkpoint 缺少 prefix 或 ImageNet 归一化参数")
    backbone = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    prefix = torch.nn.Sequential(backbone.downsample_layers[0], backbone.stages[0],
                                 backbone.downsample_layers[1], backbone.stages[1])
    prefix.load_state_dict(prefix_state, strict=True)
    return prefix.eval(), state["mean"].float(), state["std"].float(), checkpoint_path, int(
        checkpoint["epoch"])


def add_protocol_groups(pairs):
    grouped = summarize_groups(pairs)
    easy = [pair for pair in pairs
            if pair["current_visibility"] == pair["history_visibility"] == 1]
    grouped["easy_same_cell"] = summarize([
        pair for pair in easy if pair["query_cell"] == pair["true_history_cell"]
    ])
    grouped["easy_different_cell"] = summarize([
        pair for pair in easy if pair["query_cell"] != pair["true_history_cell"]
    ])
    visibility = defaultdict(list)
    for pair in pairs:
        visibility[f'current{pair["current_visibility"]}_history{pair["history_visibility"]}'].append(
            pair)
    grouped["by_endpoint_visibility"] = {
        name: summarize(group) for name, group in sorted(visibility.items())
    }
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)

    run = resolve_project_path(args.run)
    reference_path = resolve_project_path(args.reference)
    run_config = json.loads((run / "config.json").read_text())
    reference = json.loads(reference_path.read_text())
    rgb_cache = resolve_project_path(run_config["rgb_cache"])
    rgb_meta = json.loads((rgb_cache / "metadata.json").read_text())

    reference_cache = Path(reference["scope"]["cache"]).resolve()
    run_source_cache = Path(run_config["rgb_source_cache"]).resolve()
    rgb_source_cache = Path(rgb_meta["source_cache"]).resolve()
    if not reference_cache == run_source_cache == rgb_source_cache:
        raise ValueError("run、reference 与 RGB 缓存的特征来源引用不一致")
    if run_config.get("source_visual_config") != rgb_meta.get("source_config"):
        raise ValueError("run 与 RGB 缓存的视觉配置引用不一致")
    if tuple(reference["scope"]["grid_hw"]) != (36, 64):
        raise ValueError("诊断固定使用 stage1 36×64 特征格")

    frames = rgb_meta["frames"]
    frame_lookup = {
        (row["game"], row["clip"], int(row["original_frame_id"])): index
        for index, row in enumerate(frames)
    }
    rgb = np.load(rgb_cache / "rgb.npy", mmap_mode="r")
    if rgb.shape != (len(frames), 3, 288, 512) or rgb.dtype != np.uint8:
        raise ValueError("RGB 缓存必须是 N×3×288×512 uint8")

    prefix, mean, std, checkpoint_path, best_epoch = load_prefix(run)
    grid_hw = (36, 64)
    original_hw = tuple(reference["scope"].get("original_hw", (720, 1280)))
    reference_pairs = reference["pairs"]
    queries = defaultdict(list)
    for pair in reference_pairs:
        if pair["delta"] not in (1, 2):
            raise ValueError("参考只能包含固定 Δ1/Δ2 pair")
        identity = (pair["game"], pair["clip"], int(pair["current_original_frame_id"]))
        queries[identity].append(pair)

    started = time.perf_counter()
    forward_seconds = 0.
    pairs = []
    with torch.inference_mode():
        for query_number, (current_identity, query_pairs) in enumerate(queries.items(), 1):
            identities = [current_identity]
            for pair in query_pairs:
                history_identity = (pair["game"], pair["clip"],
                                    int(pair["history_original_frame_id"]))
                if history_identity not in identities:
                    identities.append(history_identity)
            try:
                frame_indices = [frame_lookup[identity] for identity in identities]
            except KeyError as error:
                raise ValueError(f"参考帧不在 run 的 RGB 缓存中: {error.args[0]}") from error

            pixels = torch.from_numpy(rgb[frame_indices]).float()
            forward_started = time.perf_counter()
            features = prefix((pixels / 255 - mean) / std)
            forward_seconds += time.perf_counter() - forward_started
            if features.shape != (len(identities), 192, *grid_hw) or not torch.isfinite(features).all():
                raise ValueError(f"prefix 输出非法: {tuple(features.shape)}")
            feature_by_identity = dict(zip(identities, features))

            for reference_pair in query_pairs:
                history_identity = (reference_pair["game"], reference_pair["clip"],
                                    int(reference_pair["history_original_frame_id"]))
                pair = {key: value for key, value in reference_pair.items() if key != "results"}
                pair["results"] = analyze_pair(
                    feature_by_identity[current_identity], feature_by_identity[history_identity],
                    int(pair["query_cell"]), int(pair["true_history_cell"]),
                    pair["history_gt_xy"], grid_hw, original_hw)
                pairs.append(pair)

            if query_number % 25 == 0 or query_number == len(queries):
                print(json.dumps({"queries": query_number, "total_queries": len(queries),
                                  "pairs": len(pairs),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)

    aggregate = {
        str(delta): add_protocol_groups([pair for pair in pairs if pair["delta"] == delta])
        for delta in (1, 2)
    }
    elapsed = time.perf_counter() - started
    output = {
        "scope": {
            "diagnostic": "adapted stage1 feature correspondence with a current-GT oracle query",
            "limitation": "This is conditional matching, not automatic detection or a new motion module.",
            "visibility_limitation": "VC3 groups measure response at an annotated coordinate; they are not positive visual correspondence evidence.",
            "primary": "both endpoints VC1: spatial_mean_centered radius2 for delta1, radius4 for delta2, plus global recall@1/5/10; same/different native cells reported separately",
            "grid_hw": list(grid_hw),
            "original_hw": list(original_hw),
            "device": "cpu",
            "precision": "float32 online prefix features",
            "torch_threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine(),
            "torch_version": str(torch.__version__),
            "run": str(run.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "reference": str(reference_path.resolve()),
            "rgb_cache": str(rgb_cache.resolve()),
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "run_config": run_config,
        "best_epoch": best_epoch,
        "counts": {
            "queries": len(queries),
            "pairs": len(pairs),
            "eligible_pairs": {str(delta): sum(pair["delta"] == delta for pair in pairs)
                               for delta in (1, 2)},
        },
        "reference_counts": reference.get("counts"),
        "aggregate": aggregate,
        "pairs": pairs,
        "elapsed_seconds": elapsed,
        "prefix_forward_seconds": forward_seconds,
        "timing_scope": "RGB window indexing + CPU float32 prefix forward + matching + aggregation; excludes imports, initial metadata/checkpoint loading, model construction and final JSON write",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
