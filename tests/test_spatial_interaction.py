"""同预算读出须保留非线性位置这一实际区别，且兼容原定位/缺失输出。"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ballmotion import probe


def check():
    assert hasattr(probe, 'SpatialInteractionReadout'), '尚无可比较的跨地址读出'
    cls = probe.SpatialInteractionReadout
    torch.manual_seed(4)
    same = cls(2, 3, 1, 'same_address').double()
    torch.manual_seed(4)
    cross = cls(2, 3, 1, 'cross_address').double()
    for key, value in same.state_dict().items():
        torch.testing.assert_close(value, cross.state_dict()[key], rtol=0, atol=0)
    assert sum(p.numel() for p in same.parameters()) == sum(p.numel() for p in cross.parameters())
    for interaction in ('same_address', 'cross_address'):
        actual = cls(576, 32, 8, interaction)
        assert sum(p.numel() for p in actual.parameters()) == 30880
        assert sum(m.weight.numel() for m in actual.modules()
                   if isinstance(m, torch.nn.Conv2d)) * 36 * 64 == 70778880
    # 构造确定的非零跨地址二阶导数；norm之前的原图不在这个命题内。
    for model in (same, cross):
        for name, param in model.named_parameters():
            torch.nn.init.constant_(param, .1 if name.endswith('weight') else 0.)
    mixed = []
    for model in (same, cross):
        x = torch.full((1, 2, 3, 3), .2, dtype=torch.float64, requires_grad=True)
        y = model(x)[0, 0, 1, 1]
        first = torch.autograd.grad(y, x, create_graph=True)[0][0, 0, 1, 0]
        second = torch.autograd.grad(first, x)[0][0, 0, 1, 2]
        mixed.append(float(second))
    assert mixed[0] == 0. and abs(mixed[1]) > 1e-8, mixed

    features = torch.randn(2, 6, 5, 7)
    outputs = []
    for interaction in ('same_address', 'cross_address'):
        model = probe.SpatialProbe(6, upscale=2, hidden_channels=3, num_frames=3)
        model.location = cls(6, 3, 2, interaction)
        outputs.append(model)
    outputs[1].load_state_dict(outputs[0].state_dict())
    a, b = (m(features) for m in outputs)
    assert a.shape == b.shape == (2, 141)
    torch.testing.assert_close(a[:, -1], b[:, -1], rtol=0, atol=0)
    for value, model in zip((a, b), outputs):
        torch.nn.functional.cross_entropy(value, torch.tensor([17, 140])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS: matched parameters, distinct mixed derivative, unchanged absence formula, V1/V0 gradients')


if __name__ == '__main__':
    check()
