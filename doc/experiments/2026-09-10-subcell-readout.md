# 统一输出网格后，中层特征能否读出更细球位置？

日期：2026-09-10。状态：线性与非线性读出比较已完成。
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

两次线性子格运行的代码版本为 `84b25e4`。实际结果：

| 层 / 读出 | train PCK@8 | val PCK@8 | val PCK@16 | val PCK@32 | val 中位误差 | 最佳 epoch | 秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stage1 原生（对照） | 34.86% | 33.93% | 61.61% | 63.39% | 10.12 | 18 | 77.76 |
| stage1 子格 | 51.49% | 51.79% | 61.61% | 61.61% | 7.45 | 30 | 65.78 |
| stage2 原生（对照） | 9.74% | 8.48% | 40.18% | 73.66% | 18.43 | 20 | 47.13 |
| stage2 子格 | 27.57% | 10.27% | 39.73% | 65.63% | 21.27 | 24 | 36.82 |

**stage1 的严格定位确实受原生读出限制。** 同一特征缓存，train/val 的 PCK@8 都提升；验证增加 17.86 个百分点，而 PCK@16 未变。可以说通道中存在能被该头利用的细位置信息，不能说新增了视觉证据。最佳 epoch 在运行末尾，还不能声称已经收敛。

**stage2 没有同样的验证收益。** 训练严格定位改善，验证只小幅变化且宽容差指标下降。它既可能受读出容量/优化限制，也可能出现细相位泛化问题，不能直接判为信息不可恢复。两者都仍存在大量背景误选；本轮没有加入 motion，也没有证实时序救回。

耗时因头结构和删除重复 batch 复制而变化，不作为控制充分的性能优化比较。各运行的完整 train/val、分组、存在指标与逐帧预测保存在对应输出目录。

## 下一对照：固定宽度非线性头

在相同 stage、stride-4 输出、缓存和训练条件下，空间头改为 `1×1(C,32) -> GELU -> 3×3(32,r²) -> PixelShuffle`。无球分支形式不变。stage1/2 参数分别为 7,525 / 17,329；这是读出容量对照，不作新颖性主张，不增加额外 backbone/RGB/motion 模块。训练仍为 30 epochs，先看现有冻结特征是否能被更充分利用。

```bash
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8 --output outputs/spatial_probe/nonlinear_stage1_seed0 --stage 1 --output-stride 4 --hidden-channels 32
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8 --output outputs/spatial_probe/nonlinear_stage2_seed0 --stage 2 --output-stride 4 --hidden-channels 32
```

新增非线性梯度/子格拟合测试，当前共 8 项；真实三轮 smoke 通过后启动完整运行。若改善明显，保留更强空间头再研究 temporal；若仍差，下一步比较高输入分辨率与适度微调，避免把弱头当成 motion 对照。

## 非线性头实际结果与决定

运行代码 `c1652db`；相同缓存、30 epoch。以下均是各自最佳验证 epoch，未混用最后一个 epoch。

| 层 | 最佳 epoch | train PCK@8 / @16 | val PCK@8 / @16 / @32 | val 中位误差 | 秒数 |
|---|---:|---|---|---:|---:|
| stage1 | 3 | 59.58% / 75.94% | 59.38% / 76.79% / 78.57% | 6.52 | 67.96 |
| stage2 | 7 | 45.86% / 62.16% | 20.09% / 53.13% / 75.00% | 15.51 | 41.44 |

stage1 比同网格线性头的 val PCK@16 高 15.18 个百分点；这明确表明首轮弱头限制了可读性。stage2 也有收益，但严格位置读出依然较差。最佳 epoch 很早，后续损失下降不能被解释为泛化持续改善。无球分支仍全部预测有球，问题未解决。

下一步保留这个更强空间头，比较更高输入尺度，并让最终输出网格在原图坐标中一致。暂不引入时序模块；否则把弱空间头当对照会夸大 motion 的贡献。随后仍需扩大开发训练数据、验证训练稳定性并解决存在判别，才能将其作为完整检测基线。见[输入尺度实验](2026-09-10-input-scale.md)。
