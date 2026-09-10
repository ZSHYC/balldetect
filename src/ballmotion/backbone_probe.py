"""同一读出头下，比较冻结视觉前缀与共同微调。"""
import torch
from torch import nn


class BackboneProbe(nn.Module):
    def __init__(self, prefix, head, train_backbone):
        super().__init__()
        self.prefix = prefix.requires_grad_(train_backbone)
        self.head = head
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def encode(self, pixels):
        return self.prefix((pixels.float() / 255 - self.mean) / self.std)

    def forward(self, pixels):
        # 输入为B,T,RGB,H,W；逐帧前缀共享参数，随后恢复原始时间顺序。
        b, t = pixels.shape[:2]
        features = self.encode(pixels.flatten(0, 1))
        return self.head(features.reshape(b, t * features.shape[1], *features.shape[-2:]))
