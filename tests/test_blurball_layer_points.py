"""源像素中心映射与官方原生层接口必须一致。"""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_blurball_layer_points import ConvNeXt, official_layers, sample_layer_points


def test_known_feature_centers_and_point_permutation():
    features = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    xy = torch.tensor([[[1.5, 1.5], [5.5, 1.5], [3.5, 3.5]]])
    wh = torch.tensor([[8., 8.]])
    normalized = F.group_norm(features, 1)
    expected = torch.stack((normalized[0, :, 0, 0], normalized[0, :, 0, 1],
                            normalized[0].mean((-2, -1))), dim=0)[None]
    result = sample_layer_points(features, xy, wh)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(sample_layer_points(features, xy[:, [2, 0, 1]], wh),
                                result[:, [2, 0, 1]])


def test_layer_extraction_matches_upstream_without_resizing():
    # 小尺寸同架构仅检查层位/布局，预训练权重的读取由真实预检确认。
    torch.set_num_threads(2)
    model = ConvNeXt(depths=[1, 1, 1, 1], dims=[4, 8, 16, 32]).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(.1)
    x = torch.randn(2, 3, 32, 64)
    with torch.inference_mode():
        actual = official_layers(model, x)
        expected = model.get_intermediate_layers(x, n=[1, 2], reshape=True, norm=False)
    for name, reference in zip(('pretrained_stage1', 'pretrained_stage2'), expected):
        torch.testing.assert_close(actual[name], reference, rtol=0, atol=0)
