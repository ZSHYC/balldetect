"""共同坐标协议下的二值盘、WASB概率QFL与原始热图读出。"""
import torch

from ballmotion.tennis import grid_to_original


def disk_targets(target_indices, grid_hw):
    height, width = grid_hw
    indices = target_indices[:, None, None]
    x = torch.arange(width, device=indices.device)[None, None, :]
    y = torch.arange(height, device=indices.device)[None, :, None]
    squared_distance = (x - indices % width).square() + (y - indices // width).square()
    return ((squared_distance <= 2.5 ** 2) & (indices < height * width)).float().unsqueeze(1)


def quality_focal_loss(logits, targets):
    probability = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    log_likelihood = targets * probability.log() + (1 - targets) * (1 - probability).log()
    return -((probability - targets).square() * log_likelihood).mean()


@torch.no_grad()
def heatmap_predictions(logits):
    peak, indices = logits.flatten(1).max(1)
    xy = grid_to_original(indices.cpu().numpy(), logits.shape[-2:])
    return xy, peak.sigmoid().cpu().numpy()
