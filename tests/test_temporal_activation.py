"""目标/支持拆分只改变GELU与求和顺序，保留原参数和时间槽语义。"""
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ballmotion.probe import SpatialInteractionReadout, SpatialProbe


def heads(slot=2):
    torch.manual_seed(7)
    early = SpatialInteractionReadout(10, 3, 2, 'cross_address', num_frames=5).double()
    torch.manual_seed(7)
    separate = SpatialInteractionReadout(
        10, 3, 2, 'target_activation', num_frames=5, target_slot=slot).double()
    return early, separate


def test_identical_parameters_and_linear_limit_including_gradients():
    early, separate = heads()
    assert early.state_dict().keys() == separate.state_dict().keys()
    for key, value in early.state_dict().items():
        torch.testing.assert_close(value, separate.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 10, 4, 5, dtype=torch.float64)
    with patch('torch.nn.functional.gelu', side_effect=lambda value: value):
        before, after = early(x), separate(x)
        torch.testing.assert_close(before, after, rtol=1e-12, atol=1e-12)
        before.square().sum().backward()
        after.square().sum().backward()
    for a, b in zip(early.parameters(), separate.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('slot', [0, 2, 4])
def test_two_path_result_matches_explicit_frame_contributions(slot):
    early, model = heads(slot)
    x = torch.randn(2, 10, 4, 5, dtype=torch.float64, requires_grad=True)
    contributions = [F.conv2d(x[:, 2*i:2*i+2], model.project.weight[:, 2*i:2*i+2],
                              model.project.bias / 5) for i in range(5)]
    merged = F.gelu(contributions[slot]) + F.gelu(sum(
        contribution for i, contribution in enumerate(contributions) if i != slot))
    expected = model.readout(F.gelu(model.spatial(model.address(merged))))
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(actual, early(x), atol=1e-8, rtol=1e-8)
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(x.grad[:, i*2:(i+1)*2].abs().sum() > 0 for i in range(5))


def test_absence_logit_unchanged_and_visible_absent_targets_backpropagate():
    early, separate = heads()
    models = []
    for location in (early, separate):
        head = SpatialProbe(10, upscale=2, hidden_channels=3, num_frames=5).double()
        head.location = location
        models.append(head)
    models[1].load_state_dict(models[0].state_dict())
    x = torch.randn(2, 10, 4, 5, dtype=torch.float64)
    a, b = (model(x) for model in models)
    assert a.shape == b.shape == (2, 81)
    torch.testing.assert_close(a[:, -1], b[:, -1], rtol=0, atol=0)
    F.cross_entropy(b, torch.tensor([7, 80])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in models[1].parameters())


def test_missing_target_slot_is_not_silently_assumed_to_be_last():
    with pytest.raises(ValueError, match='明确目标槽'):
        SpatialInteractionReadout(10, 3, 2, 'target_activation', num_frames=5)
