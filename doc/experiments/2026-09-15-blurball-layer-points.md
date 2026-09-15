# 较深层是否补充真实候选的球身份信息

日期：2026-09-15。状态：特征提取运行中。
协议：[冻结层间点特征v1](../protocols/blurball-layer-points-v1.md)。

## 为什么现在做

目标/支持独立激活已完成12轮并未通过保留条件，继续沿第一处GELU扫描没有依据。现代检测器文献提示深层特征可能有身份价值，但本模型只有DINOv3浅层；先用相同候选测试这一缺口，比直接增加teacher或深层融合网络更有区分力。

本轮不是把未验证的GT可分性直接变成架构理由。主问题是自动候选残差排序能否实际减少错误，三个来源各自训练相同形式读出；冻结点特征的负结果也不能扩大为整个视觉表征无效。

## 已有训练候选与难度

增强center5 best8训练候选已导出，37590个目标，K1及q复现；提取耗时279.47秒，峰值显存451.52MiB。验证候选复用13912个目标，未重复模型前向。路径为`outputs/blurball/centered_hflip/center5_seed0/{train_candidates,candidate_coverage}/`。

| V1条件 | train | val |
| --- | ---: | ---: |
| 全部可见目标 | 32134 | 12689 |
| 原K1错误≥4px | 1213 | 1876 |
| K16无4px正确位置 | 704 | 1228 |
| 原K1错但K16可救回 | 509 | 648 |
| 原K1距GT≥max(16,l+4) | 229 | 1348 |

两侧每个V1都有至少一个远负候选。训练侧31430个eligible中，30921个原K1正确、509个可救回；可救回组109个原K1是远负，400个原K1在忽略区。验证对应11461个eligible、10813个原K1正确、648个可救回（515远负、133忽略区）。229个训练远负K1不是229个可救回正负对。精确计数见`outputs/blurball/layer_points/supervision_counts.json`。

训练候选较容易是实测分布差异，不是数据泄漏结论，也不自动要求重新训练多折候选生成器。更不能从帧数量直接推导难例梯度被淹没：本次masked loss的零残差起点，recovery已经占总loss83.48%、逐目标logit梯度L1总量62.36%。后者不等于共享参数合成梯度，但足以否定“数量少所以必须强重加权”的简单理由。

因此在正式训练前取消最初拟议的保留/救回各0.5，最终使用普通eligible帧均值。这个更正不是看层间验证结果后改超参。`initial_loss_mass.json`保留起点数值，计算使用已有peak logits，不占用GPU或读取RGB。

## 实现与预检

`scripts/export_blurball_layer_points.py`提取官方stage1、官方stage2、适配stage1的目标帧点特征。`tests/test_blurball_layer_points.py`的像素中心/点排列与官方层接口测试已通过2项。

真实预检使用训练侧前4个目标（缓存帧ID4–7），成功读取完整官方权重与适配best8。原生浅层特征形状4×192×36×64，深层4×384×18×32，各点描述符数值有限。适配提取与实际`encode`输出完全一致，独立GroupNorm与原五帧头的目标组也完全一致。峰值已分配显存288.90MiB，是B4功能预检，不是正式提取资源上界。证据为`outputs/blurball/layer_points/preflight.{py,json}`。

使用Conda `zshihyc`与共享RTX5070Ti Laptop。源码ebd445f的特征导出已启动，实际命令：

```bash
/home/zshyc/miniforge3/envs/zshihyc/bin/python -u scripts/export_blurball_layer_points.py \
  --output outputs/blurball/layer_points/features --batch-size 16
```

日志位于`outputs/blurball/layer_points/export.log`。GT锚点列与自动候选列严格分开使用。正式探针尚未运行，当前没有层间性能结论。

读出入口为[train_blurball_layer_points.py](../../scripts/train_blurball_layer_points.py)，GT隔离、零初始化、ambiguous零梯度和eligible等权四项测试通过。真实首256个训练目标包含245个V1、238个eligible（232保留、6救回），三种描述符的零残差初始loss均为0.09471084，原分数精确保持，梯度与一次AdamW更新均有限。证据为`readout_preflight.{py,json}`，预检不保存训练权重、不改变正式seed。

正式训练每轮完整遍历37590个目标，原q固定，三层分别30轮last评价；最后一轮验证预测直接复用，另作一次固定last的完整训练集预测，以区分训练拟合与未见比赛效果。不会为同一验证权重重复前向。
