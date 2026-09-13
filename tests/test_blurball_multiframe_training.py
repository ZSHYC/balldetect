"""五帧上下文训练必须共享目标，并把监督对准显式的 t 槽位。"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "scripts/train_blurball_midpoint.py"
spec = importlib.util.spec_from_file_location("train_blurball_midpoint", SCRIPT)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def _frames():
    return [
        dict(match="00", rally=rally, game="match00", clip=rally,
             original_frame_id=frame_id, split="train")
        for rally in ("001", "002") for frame_id in range(7)
    ]


def test_five_frame_arms_share_targets_and_center_on_t():
    frames = _frames()
    cached_windows = np.asarray([
        [base + frame_id - 2, base + frame_id - 1, base + frame_id]
        for base in (0, 7) for frame_id in range(2, 7)
    ])
    boundaries = [dict(match="00", rally="001", new_segment_start="3")]

    causal, causal_rows, causal_slot, causal_info = trainer.prepare_training_windows(
        frames, cached_windows, boundaries, "causal5", "history")
    centered, centered_rows, centered_slot, centered_info = trainer.prepare_training_windows(
        frames, cached_windows, boundaries, "center5", "history")

    assert causal.tolist() == [[7, 8, 9, 10, 11]]
    assert centered.tolist() == [[9, 10, 11, 12, 13]]
    assert causal_slot == 4
    assert centered_slot == 2
    assert [row["original_frame_id"] for row in causal_rows] == [4]
    assert [row["original_frame_id"] for row in centered_rows] == [4]
    assert causal_info["base_targets"] == centered_info["base_targets"] == {
        "all": 10, "train": 10}
    assert causal_info["natural_targets"] == centered_info["natural_targets"] == {
        "causal5": {"all": 3, "train": 3},
        "center5": {"all": 3, "train": 3},
    }
    assert causal_info["common_targets"] == centered_info["common_targets"] == {
        "all": 1, "train": 1}
    for arm in ("causal5", "center5"):
        np.testing.assert_array_equal(causal_info["excluded_target_indices"][arm],
                                      centered_info["excluded_target_indices"][arm])

    repeated, rows, target_slot, _ = trainer.prepare_training_windows(
        frames, cached_windows, boundaries, "center5", "repeat_current")
    assert repeated.tolist() == [[11, 11, 11, 11, 11]]
    assert rows == centered_rows
    assert target_slot == 2


def test_default_training_windows_remain_causal_three_frame():
    frames = _frames()
    cached_windows = np.asarray([[0, 1, 2], [1, 2, 3], [7, 8, 9]])
    boundaries = [dict(match="00", rally="001", new_segment_start="3")]

    windows, rows, target_slot, info = trainer.prepare_training_windows(
        frames, cached_windows, boundaries, None, "history")

    assert windows.tolist() == [[0, 1, 2], [7, 8, 9]]
    assert [row["original_frame_id"] for row in rows] == [2, 2]
    assert target_slot == 2
    assert info["excluded_target_indices"].tolist() == [3]


@pytest.mark.parametrize('extra', [[], ['--interaction', 'cross_address',
                                      '--temporal-input', 'repeat_current']])
def test_other_inputs_cannot_claim_five_frame_context_protocol(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['train_blurball_midpoint.py', '--window', 'center5',
                                     '--rgb-cache', 'unused', '--output', 'unused', *extra])
    with pytest.raises(SystemExit) as raised:
        trainer.main()
    assert raised.value.code == 2


def test_five_frame_cross_address_model_has_real_forward_and_gradients():
    weights = ROOT / ("models/pretrained/dinov3/lvd1689m/"
                      "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth")
    model = trainer.build_dino_model(
        weights, upscale=2, interaction="cross_address", num_frames=5)
    pixels = torch.randint(0, 256, (1, 5, 3, 32, 32), dtype=torch.uint8)

    logits = model(pixels)
    logits.sum().backward()

    assert logits.shape == (1, 8 * 8 + 1)
    assert model.head.location.project.in_channels == 192 * 5
    assert model.head.norm.num_groups == 5
    assert all(parameter.grad is not None for parameter in model.head.parameters())
