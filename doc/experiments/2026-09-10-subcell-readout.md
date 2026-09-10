# 统一输出网格后，中层特征能否读出更细球位置？

日期：2026-09-10。状态：准备运行。
沿用 [v1 协议](../protocols/tennis-spatial-probe-v1.md) 的数据、坐标、单帧输入、标签、指标与 train/val 划分；本轮明确改变读出网格，结果与原生网格分列，不覆盖首轮。

## 为什么进行这轮

[首轮结果](2026-09-10-spatial-probe.md)中 stride-16 层的 PCK@32 最好，但严格容差被网格量化限制。这有两种解释：中层保留了更强球识别和细位置内容，只是头没有读出；或者只保留粗位置，细坐标需要额外 RGB 或更早层。先检验前一种，不预设细节分支必需。

## 唯一结构变化

仍对单个冻结 stage 作 GroupNorm。空间 1×1 Conv 从输出 1 通道改为输出 `r²` 通道，再 PixelShuffle(r)；每个通道对应原生 cell 内一个子格，输出统一 stride 4。stage 1 的 r=2、参数 965；stage 2 的 r=4、参数 6,545。无球头形式不变，类别数偏置按最终输出网格计算，训练仍为 H×W+1 类交叉熵。

这是常规子像素读出，不声称模型创新；没有添加新 RGB 观测或时序。它确实增加参数并改变监督的细粒度，不能称纯插值或完全等容量。和原生头相比，只有同一 stage 的变化可用于判断细位置读出潜力，不能仅据此比较跨层容量。

使用相同 1,559/239 帧缓存、seed=0、lr=.003、batch=16、30 epoch、AdamW(.01)。空间网格和已知 cell 内位置样例的 7 个测试已通过，stage1 三轮真实 smoke 完成。去掉 NumPy 高级索引之后重复的 `.copy()`：高级索引本来就分配 batch；减少内存复制，不改变输入数值。

```bash
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8 --output outputs/spatial_probe/subcell_stage1_seed0 --stage 1 --output-stride 4
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8 --output outputs/spatial_probe/subcell_stage2_seed0 --stage 2 --output-stride 4
```

## 结果与后续

尚未完成实际比较，不填预期分数。若 train 与 val 的严格位置指标都明显改善，可说明对应细信息能被该读出利用；若只有 train 改善，先考虑容量和泛化；若仍不能拟合，继续非线性读出和尺度对照，不立即宣布特征没有球信息。
