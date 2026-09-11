"""固定细节读出的槽位、低通位置和零起点/梯度检查。"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_tennis_detail_readout import DetailResidual, current_features


def test_frozen_detail_readout():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    base = nn.Module()
    base.prefix = nn.Sequential(nn.Conv2d(3, 96, 1), nn.GELU())
    base.register_buffer("mean", torch.zeros(1, 3, 1, 1))
    base.register_buffer("std", torch.ones(1, 3, 1, 1))
    base.requires_grad_(False).eval()
    rgb = np.random.default_rng(0).integers(0, 256, (5, 3, 6, 8), dtype=np.uint8)
    windows = np.array([[0, 1, 4], [1, 2, 3]])
    ids = np.array([1, 0])
    features = current_features(base, rgb, windows, ids, torch.device("cpu"))
    expected = base.prefix(torch.from_numpy(rgb[[3, 4]]).float() / 255)
    assert torch.equal(features, expected) and not features.requires_grad

    native, pooled = DetailResidual("native"), DetailResidual("pooled")
    pooled.load_state_dict(native.state_dict())
    logits = torch.randn(2, 49)
    assert sum(p.numel() for p in native.parameters()) == 3392
    assert torch.equal(native(features, logits), logits)
    assert torch.equal(pooled(features, logits), logits)
    optimizer = torch.optim.AdamW(native.parameters(), lr=3e-4, weight_decay=.01)
    loss = F.cross_entropy(native(features, logits), torch.tensor([10, 48]))
    loss.backward()
    assert native.location[2].weight.grad.abs().sum() > 0
    assert native.location[0].weight.grad.count_nonzero() == 0
    assert all(p.grad is None for p in base.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = native(features, logits)
    assert torch.equal(output[:, -1], logits[:, -1])
    assert not torch.equal(output[:, :-1], logits[:, :-1])
    F.cross_entropy(output, torch.tensor([10, 48])).backward()
    assert native.location[0].weight.grad.abs().sum() > 0

    pooled.load_state_dict(native.state_dict())
    normalized = F.group_norm(features, 1)
    lowpass = F.interpolate(F.avg_pool2d(normalized, 2), size=(6, 8),
                           mode="bilinear", align_corners=False)
    expected_residual = pooled.location(lowpass).flatten(1)
    torch.testing.assert_close(pooled(features, logits)[:, :-1],
                               logits[:, :-1] + expected_residual, rtol=0, atol=0)
    assert not torch.equal(native(features, logits), pooled(features, logits))
    assert all(p.grad is None for p in base.parameters())


if __name__ == "__main__":
    test_frozen_detail_readout()
    print("frozen detail readout check passed")
