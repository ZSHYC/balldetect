"""固定训练batch的端点辅助尺度校准与一次真实更新；不使用验证标签。"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from train_tennis_heatmap import ROOT, build_dino_model, model_input
from ballmotion.correspondence import endpoint_logits, endpoint_loss, native_endpoint_cells
from ballmotion.tennis import grid_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    reference = ROOT / "outputs/full_heatmap/dino_prefix_seed0/config.json"
    config = json.loads(reference.read_text())
    cache = ROOT / config["rgb_cache"]
    metadata = json.loads((cache / "metadata.json").read_text())
    frames = metadata["frames"]
    windows = np.asarray(metadata["windows"])
    cells = native_endpoint_cells(frames, windows)
    train_idx = np.asarray([i for i, w in enumerate(windows) if frames[w[-1]]["game"] != "game7"])
    # 只按原训练顺序选第一个两种Δ均有合法pair的batch，不看任何模型/验证分数。
    for start in range(0, len(train_idx), 8):
        ids = train_idx[start:start + 8]
        batch_cells = cells[ids]
        current = batch_cells[:, 2]
        counts = []
        for delta, radius in ((1, 2), (2, 4)):
            history = batch_cells[:, 2 - delta]
            valid = ((current < 2304) & (history < 2304)
                     & ((history % 64 - current % 64).abs() <= radius)
                     & ((history // 64 - current // 64).abs() <= radius))
            counts.append(int(valid.sum()))
        if min(counts) > 0:
            break
    else:
        raise ValueError("训练数据没有满足校准条件的batch")

    model = build_dino_model(config["weights"])
    initial = torch.load(ROOT / "outputs/full_heatmap/dino_smoke_seed0/best.pt",
                         map_location="cpu", weights_only=True)
    assert initial["epoch"] == 0
    assert model.state_dict().keys() == initial["model"].keys()
    assert all(torch.equal(value, initial["model"][name]) for name, value in model.state_dict().items())
    rng_before = torch.get_rng_state()
    query = torch.nn.Parameter(F.normalize(torch.randn(192, generator=torch.Generator().manual_seed(0)),
                                            dim=0).cuda())
    assert torch.equal(rng_before, torch.get_rng_state())
    model.cuda().train()
    model.prefix.eval()
    rgb = np.load(cache / "rgb.npy", mmap_mode="r")
    pixels = model_input(rgb, windows, ids, "cuda", "dino")
    target_rows = [frames[windows[i, -1]] for i in ids]
    target = torch.from_numpy(grid_targets(
        [[r["x_raw"], r["y_raw"]] for r in target_rows],
        [r["visibility_raw"] != 0 for r in target_rows], (72, 128))).cuda()
    batch_cells = batch_cells.cuda()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        direct = model(pixels)
    features = model.encode(pixels.flatten(0, 1)).reshape(len(ids), 3, 192, 36, 64)
    logits = model.head(features.flatten(1, 2))
    assert torch.equal(direct, logits)
    main_loss = F.cross_entropy(logits, target)
    relation_pairs = endpoint_logits(features, batch_cells)
    assert [len(pair["target"]) for pair in relation_pairs] == counts
    relation_loss = endpoint_loss(relation_pairs)
    appearance_loss = endpoint_loss(endpoint_logits(features, batch_cells, query))
    norms = {}
    for name, loss in (("main", main_loss), ("relation", relation_loss), ("appearance", appearance_loss)):
        gradients = torch.autograd.grad(loss, tuple(model.prefix.parameters()), retain_graph=True)
        norm = torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
        assert torch.isfinite(norm) and norm > 0, name
        norms[name] = float(norm)
        del gradients
    appearance_weight = .1 * norms["relation"] / norms["appearance"]
    result = {
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "reference_config": str(reference), "seed": 0, "temperature": .1,
        "batch_targets": [[r["game"], r["clip"], r["original_frame_id"]] for r in target_rows],
        "valid_pairs_delta1_delta2": counts, "prefix_gradient_norms": norms,
        "relation_weight": .1, "appearance_weight": appearance_weight,
        "weighted_relation_to_main_gradient_norm": .1 * norms["relation"] / norms["main"],
        "initial_state_tensors_equal": len(initial["model"]), "main_logits_max_abs_difference": 0.,
        "initial_losses": {"main": float(main_loss.detach()), "relation": float(relation_loss.detach()),
                           "appearance": float(appearance_loss.detach())},
        "updates": {},
    }
    # 校准不会更新参数。两臂各从同一初始化做一次实际AdamW更新后退出。
    del features, logits, main_loss, relation_pairs, relation_loss, appearance_loss, loss
    for arm, weight in (("relation", .1), ("appearance", appearance_weight)):
        model.load_state_dict(initial["model"])
        groups = [{"params": model.head.parameters(), "lr": 3e-4},
                  {"params": model.prefix.parameters(), "lr": 1e-5}]
        if arm == "appearance":
            groups.append({"params": [query], "lr": 3e-4})
        optimizer = torch.optim.AdamW(groups, weight_decay=.01)
        features = model.encode(pixels.flatten(0, 1)).reshape(len(ids), 3, 192, 36, 64)
        main_loss = F.cross_entropy(model.head(features.flatten(1, 2)), target)
        auxiliary = endpoint_loss(endpoint_logits(features, batch_cells,
                                                   query if arm == "appearance" else None))
        total = main_loss + weight * auxiliary
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        parameters = list(model.parameters()) + ([query] if arm == "appearance" else [])
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        if arm == "appearance":
            assert query.grad.abs().sum() > 0
        optimizer.step()
        first_name, first_parameter = next(iter(model.named_parameters()))
        change = float((first_parameter.detach().cpu() - initial["model"][first_name]).abs().max())
        assert change > 0 and all(torch.isfinite(p).all() for p in parameters)
        result["updates"][arm] = {"loss": float(total.detach()), "first_prefix_parameter_max_change": change}
        del features, main_loss, auxiliary, total, optimizer, groups, parameters
    torch.cuda.synchronize()
    result.update(elapsed_seconds=time.perf_counter() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
