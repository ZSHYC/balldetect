"""局部cosine对应：冻结dense探针与训练期稀疏中心监督。"""
import numpy as np
import torch
from torch.nn import functional as F

from .tennis import grid_targets


def native_endpoint_cells(frames, windows, grid_hw=(36, 64)):
    """非VC1用网格末尾的无效哨兵；不改变主任务的VC2/3位置标签。"""
    xy = np.asarray([[row["x_raw"], row["y_raw"]] for row in frames], dtype=float)
    visible = np.asarray([row["visibility_raw"] == 1 for row in frames])
    return torch.from_numpy(grid_targets(xy, visible, grid_hw)[windows])


def endpoint_logits(features, cells, appearance_query=None, temperature=.1):
    """仅读取GT query的25/81个历史候选；cells为B×3的native格索引。"""
    batch, slots, channels, height, width = features.shape
    flat = features.reshape(batch, slots, channels, height * width)
    means = flat.mean(-1)
    current = cells[:, 2]
    current_xy = torch.stack((current % width, current // width), -1)
    pairs = []
    for delta, radius in ((1, 2), (2, 4)):
        history = cells[:, 2 - delta]
        history_xy = torch.stack((history % width, history // width), -1)
        displacement = history_xy - current_xy
        valid = ((current < height * width) & (history < height * width)
                 & (displacement.abs().amax(1) <= radius))
        ids = valid.nonzero().flatten()
        axis = torch.arange(-radius, radius + 1, device=features.device)
        dy, dx = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((dx.flatten(), dy.flatten()), -1)
        candidate_xy = current_xy[ids, None] + offsets
        in_bounds = ((candidate_xy >= 0)
                     & (candidate_xy < candidate_xy.new_tensor((width, height)))).all(-1)
        candidate_cells = (candidate_xy[..., 1] * width + candidate_xy[..., 0]).clamp(
            0, height * width - 1)
        keys = flat[ids[:, None], 2 - delta, :, candidate_cells] - means[ids, 2 - delta, None]
        keys = F.normalize(keys, dim=-1)
        if appearance_query is None:
            query = F.normalize(flat[ids, 2, :, current[ids]] - means[ids, 2], dim=-1)
        else:
            query = F.normalize(appearance_query, dim=0).expand(len(ids), -1)
        logits = (query[:, None] * keys).sum(-1).clamp(-1, 1) / temperature
        logits = logits.masked_fill(~in_bounds, -torch.inf)
        target = (displacement[ids, 1] + radius) * (2 * radius + 1) + displacement[ids, 0] + radius
        pairs.append({"logits": logits, "target": target, "batch_indices": ids,
                      "same_cell": (displacement[ids] == 0).all(-1)})
    return pairs


def endpoint_loss(pairs):
    """各非空Δ先平均，再等权；全空时保留零梯度路径。"""
    losses = [F.cross_entropy(pair["logits"], pair["target"])
              for pair in pairs if len(pair["target"])]
    return torch.stack(losses).mean() if losses else sum(pair["logits"].sum() for pair in pairs)


def cost_volume(features, variant):
    """Return delta-1 radius-2 then delta-2 radius-4 costs in row-major offset order."""
    if (features.ndim != 5 or features.shape[1:3] != (3, 192)
            or features.dtype != torch.float32):
        raise ValueError("features must be float32 Bx3x192xHxW")
    if variant not in ("raw", "centered", "self_centered"):
        raise ValueError(f"Unknown cost-volume variant: {variant}")

    if variant != "raw":
        features = features - features.mean((-2, -1), keepdim=True)
    features = F.normalize(features, dim=2)
    query = features[:, 2]
    key_slots = (2, 2) if variant == "self_centered" else (1, 0)
    batch, channels, height, width = query.shape
    costs = []
    for key_slot, radius in zip(key_slots, (2, 4)):
        size = 2 * radius + 1
        patches = F.unfold(features[:, key_slot], size, padding=radius)
        patches = patches.view(batch, channels, size * size, height, width)
        scores = torch.einsum("bchw,bckhw->bkhw", query, patches).clamp_(-1, 1)
        valid = F.unfold(torch.ones(1, 1, height, width, device=features.device),
                         size, padding=radius).view(1, size * size, height, width).bool()
        scores.masked_fill_(~valid, -2)
        costs.append(scores)
        del patches
    return torch.cat(costs, dim=1)
