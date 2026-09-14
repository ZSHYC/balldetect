"""同步翻转不能制造帧间运动，且必须先反射连续坐标再量化。"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_blurball_midpoint as training


def test_hflip_preserves_time_order_and_reflects_continuous_targets():
    rows = [dict(width=10, height=6, visibility_raw=v, x_raw=x, y_raw=y)
            for v, x, y in [(1, 2., 1.5), (1, 8., .5), (0, -1., -1.)]]
    reflected = training.horizontal_flip_targets(rows, (3, 4))
    # 第一中心恰好在原量化边界；先翻转类别5会错误地产生6，正确结果是7。
    np.testing.assert_array_equal(reflected, [7, 0, 12])
    pixels = torch.arange(3 * 3 * 1 * 3 * 4).reshape(3, 3, 1, 3, 4).to(torch.uint8)
    original = pixels.clone()
    targets = torch.tensor([5, 3, 12])
    _, result = training.horizontal_flip_batch(
        pixels, targets, torch.from_numpy(reflected), torch.tensor([True, False, True]))
    assert result.tolist() == [7, 3, 12]
    assert targets.tolist() == [5, 3, 12]
    assert pixels[0, :, 0, 0].tolist() == [[3, 2, 1, 0], [15, 14, 13, 12], [27, 26, 25, 24]]
    torch.testing.assert_close(pixels[1], original[1])
    assert pixels[2, :, 0, 0].tolist() == [[75, 74, 73, 72], [87, 86, 85, 84], [99, 98, 97, 96]]


def test_unselected_windows_are_unchanged_and_flip_mask_resumes(tmp_path):
    pixels = torch.arange(24).reshape(2, 3, 1, 1, 4).to(torch.uint8)
    original = pixels.clone()
    targets = torch.tensor([0, 4])
    _, actual = training.horizontal_flip_batch(
        pixels, targets, torch.tensor([3, 4]), torch.tensor([False, False]))
    torch.testing.assert_close(pixels, original)
    torch.testing.assert_close(actual, targets)

    torch.manual_seed(29)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    rng = np.random.default_rng(7)
    path = tmp_path / 'last.pt'
    training.save_training_state(path, model, optimizer, rng, {'epoch': 1})
    expected_mask = torch.rand(17) < .5
    expected_order = rng.permutation(17)
    training.restore_training_state(path, model, optimizer, rng)
    torch.testing.assert_close(torch.rand(17) < .5, expected_mask)
    np.testing.assert_array_equal(rng.permutation(17), expected_order)
