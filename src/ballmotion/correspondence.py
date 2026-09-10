"""Dense local cosine costs for cached three-frame frozen features."""
import torch
from torch.nn import functional as F


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
