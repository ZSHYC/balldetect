# 真实历史视觉是否比当前帧重复提供净定位收益？

日期：2026-09-10。状态：缓存及三组训练 smoke 已通过，准备完整开发比较。
协议：[Tennis 因果三帧 v1](../protocols/tennis-temporal-probe-v1.md)；最近邻与输入边界见[定向文献核对](../literature/2026-09-10-causal-baselines.md)。

## 假设与边界

现有可见错误包括背景亮点/记分牌竞争，单帧较强头也没有解决困难位置与无球判别。真实历史特征可能提供变化/短时上下文以减少误选。先比较 current、stack、repeat 三个头；如果 stack 只比最弱 current 好、但与 repeat 无差异，不能把收益归因于真实历史视觉。若 stack 的救回被新增错误抵消，同样不构成净改进。

本轮不是 correspondence 模型，不以相似度或假定轨迹作为新贡献。未加入光流、场景相机估计、跨镜处理或新的数据集。

## 样本和模型

保持目标原帧号模8采样，完整历史窗口为 `t−2,t−1,t`。实际有1,733个目标，65个缺少完整历史的片段开头目标排除；缓存需要5,199个唯一真实帧。train目标1,503，val目标230；val易辨认/难辨认/遮挡/无球分别207/8/4/11。所有三组使用同样目标，不能直接与旧239帧验证数字相减。

512×288、DINOv3 ConvNeXt-Tiny stage1冻结缓存；输出72×128格，对应原图10像素cell。hidden32非线性头、AdamW(.003,.01)、batch16、30epoch、seed0。current输入192通道、7,525参数；stack/repeat为3×192通道、20,197参数，每帧独立归一化。参数量已经由真实 smoke 的 config 确认。

旧空间缓存中1,733个当前帧可复用，新计算其余3,466帧；完成后以缓存账本实测复用数为准。完整窗口只保存索引，图像/特征不按窗口复制多份。

```bash
python scripts/cache_tennis_features.py --output data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --stages 1 --history-frames 2 --reuse-cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --output outputs/temporal_probe/current_seed0 --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input current
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --output outputs/temporal_probe/stack_seed0 --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input stack
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --output outputs/temporal_probe/repeat_seed0 --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input repeat
```

## 检查与评价

新增测试覆盖：不跨clip/不重编号稀疏帧、历史未知标签可用、时间通道顺序、重复完整当前帧而非逐通道重复、逐帧独立归一化、同时统计救回和新错误。14项测试通过。

真实缓存 smoke 为8个目标、24个唯一帧，复用8帧、仅新计算16帧；复用行与源缓存逐行一致。三组分别完成2epoch、batch4的训练/预测保存，均为train4/val4，并核对了三组验证CSV的原始目标身份一致。它们只证明通路，指标不进入正式结果。输出位于 `data/cache/tennis/temporal_smoke_512` 与 `outputs/temporal_probe/smoke_{current,stack,repeat}`。首次缓存 smoke 发现复用元信息的 list shape 不能直接写入合法 npy header，已经改为 tuple 并重新完成该 smoke；没有修改原始缓存或数据。

完整缓存计时包含旧特征读取/复制、模型加载、新帧提取及数组flush；不包含前置索引读取和最终元信息JSON写入。GPU forward时间另列，不能把增量补算时间当作从零提取所有帧的时间。复用缓存中的float16转换统计属于原缓存测量，新历史批次另做有限值检查；完整时序头的精度影响尚未测量。

每组保存训练/验证逐帧CSV、日志、最佳模型与配置。`compare_predictions.py` 从相同目标/标签的CSV直接统计8/16/32px下双方都对、救回、新增错误与双方都错，按visibility/clip分组；不重做forward。完整检测和存在结果单列，不用位置改善掩盖误检。

## 结果

尚未完成。不用计划值填结果，不将单seed开发比较当论文结论。根据净收益和困难集合变化，下一步再决定是否有必要研究对应搜索、背景竞争或更强空间对照。
