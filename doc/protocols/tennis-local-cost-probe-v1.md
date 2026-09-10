# Tennis dense局部对应定位基线

状态：2026-09-10已锁定的开发诊断协议。继承[因果三帧协议](tennis-temporal-probe-v1.md)的1,503/230个train/val目标、原始帧号、输入分辨率、标签与评价；不使用最终测试比赛。

## 要解决的问题

三帧appearance已有稳定净收益；[GT query诊断](../experiments/2026-09-10-correspondence-probe.md)又显示有限范围内有可用对应。当前只检验显式对应能否进一步改善真实自动定位，不把局部cost volume当创新。相关机制先例见[定向文献核对](../literature/2026-09-10-local-correspondence-baselines.md)。

## 固定输入与候选

冻结512×288 DINOv3 ConvNeXt-Tiny stage1，使用同片段真实 `[t−2,t−1,t]`。对**全部**36×64当前格计算对应；不存在GT query、visibility筛选、预测候选top-k或硬argmax位移。无球和困难目标继续进入正常训练/验证。

前25个cost通道对应历史`t−1`，偏移`dy,dx=-2…2`；后81个对应`t−2`，偏移`dy,dx=-4…4`。每组先dy后dx行优先排列。偏移是从当前query指向历史key的偏移，不是已经验证的前向物理速度。

合法cosine约束在[-1,1]；越界候选记为有限哨兵-2，区别于合法的零相关，并且可被float16精确保存。第一轮不额外添加mask通道。各对照共享同一边界图案。

## 三个新增对照

- `raw`：对每位置向量L2归一化，计算真实跨帧局部cosine。
- `centered`：每帧、每通道减空间均值后L2归一化，计算真实跨帧cosine。
- `self_centered`：相同去均值处理，但两个历史key都替换为当前`t`特征；保留25+81通道、相同边界、参数和相关非线性。

三个对照都保留真实三帧appearance；self仅替换额外cost分支的key。因此它控制的是显式跨帧对应增量，不是删除所有历史图像。它也不是现实世界“球没有运动”的因果反事实。

## 融合与目标保持

576通道appearance先按3个帧组独立无仿射GroupNorm。106通道cost保留原尺度，绕过GN，随后与appearance拼接进入原有hidden32的1×1→GELU→3×3→PixelShuffle2定位头。输出仍72×128格。模型总参数23,589，比既有stack多3,392；三种新增对照参数完全相同。

absence仍只读取归一化appearance的全局均值，再用同一线性层和`+log(HW)`；cost不直接进入该支路。空间logits会改变整体softmax，所以不能保证存在概率不变；仍分别报告完整检测与无球情况。损失、优化器、lr .003、weight_decay .01、batch16、30epoch和选优规则保持不变。

不先改成current+cost，因为这会同时删掉已证实有效的历史appearance。先保留输入信息，再判断correspondence是否提供增量。

## 实施与计算

源冻结特征和变换均确定，cost可以按源窗口顺序一次缓存。每种缓存为1,733×106×36×64个float16，846,480,384字节，即0.788GiB；三种约2.37GiB。源图像和backbone特征不重复落盘，不持久化unfold展开张量。

一次小批量算术比较在B4、192×36×64上确认shift与unfold最大差为0。R4展开法约4.51–4.96ms、峰值已分配显存1109.85MiB，逐偏移法14.74–34.97ms、32.38MiB；共享GPU条件且只测相关算子。选择batch4的展开法用于一次缓存生成，实际完整缓存另记录耗时/峰值，不将该局部速度冒充视频定位吞吐。证据为 `outputs/correspondence_probe/kernel_benchmark.json`。

先用已知向量验证时间/位移通道与边界，再用8目标smoke确认缓存、目标索引、23589参数、梯度与预测保存。新增head测试必须确认cost不被appearance GN改写、也不直接进入absence logits；旧空间和时序行为保持。

## 判断与后续条件

先固定seed0，比较三种新对照与已有stack同seed的逐帧净救回/损害。centered若同时超出stack与self_centered，才补预先固定seed1、2；若raw成为更优候选，必须补相同raw预处理的self控制后再归因。若仅self与cross共同改善，不能宣称跨帧对应有效；若都不超过stack，不继续机械叠模块。

固定训练头需比较float32计算与float16存储的cost对最终位置/存在判断的影响；描述子去均值可能改变数值敏感性，不能借用旧单帧结果宣称已验证。更改存储精度时先保留源特征条件一致，避免把多种数值变化混为一项。后续微调backbone或变化输入增强时，以上冻结缓存不再有效，必须改用相应的数据路径。
