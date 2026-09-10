# 输入采样尺度是否限制了中层小球读出？

日期：2026-09-10。状态：两层尺度比较与量化推理对照已完成。
沿用 [v1 开发划分与评价](../protocols/tennis-spatial-probe-v1.md)，仍为单帧冻结探针、seed 0、同一 1,559/239 帧。

## 问题与控制

前轮增强读出容量后 stage1 val PCK@16=76.79%，stage2=53.13%，不足以判断 DINOv3 的细空间证据是否已丢失。本轮把输入从 512×288 提至 1024×576，检查改变 backbone 之前的采样是否改善定位。

两组输出都保持 **72×128 的原图对应网格**：低输入组 output stride=4，高输入组 output stride=8。这样 GT cell 量化上限与标签目标保持相同，不把更细输出网格偷偷混入输入尺度收益。高输入组 stage1 使用原生输出（r=1），stage2 使用子格 r=2；隐藏宽度32、GELU及3×3空间头不变。头参数分别为 6,658 / 13,861，比低输入组更少；参数差异明确报告，不称完全等容量。输入尺度也会改变特征在原图中的感受野，不能把收益称为单一物理成像因果效应。

保持 AdamW、lr .003、weight_decay .01、batch16、30epoch、同一验证选优规则。训练与验证目标帧完全相同，无输入增强、无时序、无测试比赛。

## 缓存与执行

只保存 stage1/2，预期约 8.89 GiB；其余层不落盘。先用32帧确认两层真实提取、维度和选择保存，再提取完整开发采样。D盘实测可用约94GiB，足够本轮缓存；不把WSL虚拟空闲当宿主保证。

```bash
python scripts/cache_tennis_features.py --output data/cache/tennis/dinov3_convnext_tiny_1024x576_step8_s12 --height 576 --width 1024 --stages 1 2
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_1024x576_step8_s12 --output outputs/spatial_probe/nonlinear_1024_stage1_seed0 --stage 1 --output-stride 8 --hidden-channels 32
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_1024x576_step8_s12 --output outputs/spatial_probe/nonlinear_1024_stage2_seed0 --stage 2 --output-stride 8 --hidden-channels 32
```

相同原图输出格目标的映射复用已测几何函数。量化推理检查支持读取本轮两个实际训练头，无需训练无用的第0/3层。缓存 metadata 的 shapes 记录 backbone 全部输出尺寸，`saved_stages` 明确实际保存的两层；未保存层不能用于训练。

## 结果与判别

运行版本 `9566b0f`。高输入缓存一次提取耗时73.81秒，其中GPU backbone forward合计41.38秒、峰值已分配显存1480MiB；相比低输入提取33.42/9.50秒，输入成本明显增加。此处包含当时共享GPU条件，不作严格隔离的吞吐排名。

| 输入 / stage | 最佳epoch | train PCK@16 | val PCK@8 | val PCK@16 | val PCK@32 | val中位误差（px） | 训练及末次评价秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 / 1（既有） | 3 | 75.94% | 59.38% | 76.79% | 78.57% | 6.52 | 67.96 |
| 1024 / 1 | 21 | 99.07% | 62.50% | 72.77% | 74.55% | 6.52 | 356.79 |
| 512 / 2（既有） | 7 | 62.16% | 20.09% | 53.13% | 75.00% | 15.51 | 41.44 |
| 1024 / 2 | 6 | 85.82% | 52.68% | 83.93% | 87.05% | 7.78 | 182.50 |

**结论不是“输入越大越好”。** stage1 高输入拟合训练数据更强，但验证 PCK@16 下降4.02个百分点，提示泛化/场景依赖仍有实质问题。stage2则在相同原图输出格上大幅提高30.80个百分点，低输入的失败不能直接解释为该层不可恢复地丢失了球信息。输入采样、特征层位与物理感受野联合改变结果，不能把它们拆成未测得的单因素因果结论。

**保留净收益。** 在共同224个定位目标上，stage1高输入于16px容差救回17帧、损害26帧；stage2救回75帧、损害6帧。只展示被救回的样例会错误解释stage1。比较复用了预测文件，见 `outputs/spatial_probe/scale_stage1_paired.json`、`scale_stage2_paired.json`；`scripts/compare_predictions.py` 拒绝不同目标/标签/顺序的直接比较。

**精度与状态边界。** 全部239验证帧、两个固定头的float32/fp16往返对照均为0个空间argmax和0个0.5存在判断变化，最大存在概率差0.0001266，见 `outputs/spatial_probe/precision1024.json`。存在分支仍全部预测有球，因此完整检测问题尚未解决。单一验证比赛和单seed的结论仍限定为开发诊断。

## 下一步的实际选择

采用低输入stage1作为首轮时间增量的起点，保留高输入stage2作为“增加空间成本”的竞争解释；不是宣判哪个骨干层普遍最好。首个时序比较使用严格因果三帧、同一目标集合和已验证非线性头，比较current/stack/repeat。研究问题先收窄为：真实历史视觉信息能否减少背景误选与困难标签失败，而不只是通过更强头或更高输入取得收益。协议见[因果三帧开发协议](../protocols/tennis-temporal-probe-v1.md)。
