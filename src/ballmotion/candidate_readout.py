"""固定候选分数上的小型可学习残差，不改变坐标或q。"""
import math

import torch
from torch import nn


class CandidateResidualReadout(nn.Module):
    def __init__(self, channels=96, hidden=32):
        super().__init__()
        self.scale = math.sqrt(channels)
        self.layers = nn.Sequential(nn.Linear(3*channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, query, history, peak_logits):
        inputs = torch.cat((query, history[:, 0], history[:, 1]), dim=-1)*self.scale
        return peak_logits-peak_logits[:, :1]+self.layers(inputs).squeeze(-1)


def candidate_targets(xy, gt):
    """4px固定尺度的候选软标签；可监督帧筛选由训练入口负责。"""
    distance2 = (xy-gt[:, None]).square().sum(-1)
    return (-(distance2-distance2.min(-1, keepdim=True).values)/32).softmax(-1)


def candidate_set_loss(logits, positive):
    """负对数可接受候选集合概率；调用方仅传入有正例的可见帧。"""
    scores = logits-logits.max(dim=1, keepdim=True).values
    return (scores.logsumexp(1)-scores.masked_fill(~positive, -torch.inf).logsumexp(1)).mean()
