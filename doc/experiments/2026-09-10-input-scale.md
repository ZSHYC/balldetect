# 输入采样尺度是否限制了中层小球读出？

日期：2026-09-10。状态：运行中；32帧选择层缓存与单轮真实训练 smoke 已通过。
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

尚未完成。若高输入显著改善，先选可承受的空间起点，再研究时序；若仅训练改善，优先处理泛化/数据量；若收益很小，不能继续盲目放大输入。记录提取与训练成本，选取质量和计算折中，不宣称最大输入天然最优。
