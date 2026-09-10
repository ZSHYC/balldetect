# 真实历史视觉是否比当前帧重复提供净定位收益？

日期：2026-09-10。状态：seed0/1/2三组共九次30epoch完成；seed0完整固定头精度对照已完成。
协议：[Tennis 因果三帧 v1](../protocols/tennis-temporal-probe-v1.md)；最近邻与输入边界见[定向文献核对](../literature/2026-09-10-causal-baselines.md)。

## 假设与边界

现有可见错误包括背景亮点/记分牌竞争，单帧较强头也没有解决困难位置与无球判别。真实历史特征可能提供变化/短时上下文以减少误选。先比较 current、stack、repeat 三个头；如果 stack 只比最弱 current 好、但与 repeat 无差异，不能把收益归因于真实历史视觉。若 stack 的救回被新增错误抵消，同样不构成净改进。

本轮不是 correspondence 模型，不以相似度或假定轨迹作为新贡献。未加入光流、场景相机估计、跨镜处理或新的数据集。

## 样本和模型

保持目标原帧号模8采样，完整历史窗口为 `t−2,t−1,t`。实际有1,733个目标，65个缺少完整历史的片段开头目标排除；缓存需要5,199个唯一真实帧。train目标1,503，val目标230；val易辨认/难辨认/遮挡/无球分别207/8/4/11。所有三组使用同样目标，不能直接与旧239帧验证数字相减。

512×288、DINOv3 ConvNeXt-Tiny stage1冻结缓存；输出72×128格，对应原图10像素cell。hidden32非线性头、AdamW(.003,.01)、batch16、30epoch、seed0。current输入192通道、7,525参数；stack/repeat为3×192通道、20,197参数，每帧独立归一化。参数量已经由真实 smoke 的 config 确认。

完整缓存实测复用旧空间缓存的1,733个当前帧，仅新计算3,466个历史帧。stage1特征共5,199×192×36×64个float16，约4.28GiB；完整窗口只保存索引，图像/特征不按窗口复制多份。含复用与写入的一次准备耗时40.43秒，GPU补算forward合计18.75秒，峰值已分配显存454.26MiB。源缓存与本次新提取帧分开计数，不把增量成本当全量成本。

```bash
python scripts/cache_tennis_features.py --output data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --stages 1 --history-frames 2 --reuse-cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --output outputs/temporal_probe/current_seed0 --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input current
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --output outputs/temporal_probe/stack_seed0 --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input stack
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --output outputs/temporal_probe/repeat_seed0 --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input repeat
```

## 检查与评价

新增测试覆盖：不跨clip/不重编号稀疏帧、历史未知标签可用、时间通道顺序、重复完整当前帧而非逐通道重复、逐帧独立归一化、同时统计救回和新错误。14项测试通过。

真实缓存 smoke 为8个目标、24个唯一帧，复用8帧、仅新计算16帧；复用行与源缓存逐行一致。三组分别完成2epoch、batch4的训练/预测保存，均为train4/val4，并核对了三组验证CSV的原始目标身份一致。它们只证明通路，指标不进入正式结果。输出位于 `data/cache/tennis/temporal_smoke_512` 与 `outputs/temporal_probe/smoke_{current,stack,repeat}`。首次缓存 smoke 发现复用元信息的 list shape 不能直接写入合法 npy header，已经改为 tuple 并重新完成该 smoke；没有修改原始缓存或数据。

完整缓存计时包含旧特征读取/复制、模型加载、新帧提取及数组flush；不包含前置索引读取和最终元信息JSON写入。GPU forward时间另列，不能把增量补算时间当作从零提取所有帧的时间。复用缓存中的float16转换统计属于原缓存测量，新历史批次另做有限值检查；固定时序头精度对照见下文。

每组保存训练/验证逐帧CSV、日志、最佳模型与配置。`compare_predictions.py` 从相同目标/标签的CSV直接统计8/16/32px下双方都对、救回、新增错误与双方都错，按visibility/clip分组；不重做forward。完整检测和存在结果单列，不用位置改善掩盖误检。

## 结果

三组运行代码均为 `9e8b271`（精度检查脚本的后续局部修改不影响训练）。结果均在相同230个验证目标、其中219个有位置目标上计算；train指标来自各自验证选中的checkpoint。

| 输入 | 最佳epoch | train PCK@16 | val PCK@8 | val PCK@16 | val PCK@32 | val中位/平均误差px | 检测F1@16 | 训练及末次评价秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 3 | 77.58% | 62.56% | 77.17% | 79.45% | 6.04 / 85.53 | 75.28% | 66.26 |
| stack | 8 | 96.63% | 64.84% | 83.56% | 85.39% | 6.04 / 55.61 | 81.70% | 267.84 |
| repeat | 1 | 71.94% | 55.25% | 75.34% | 76.26% | 7.11 / 109.90 | 73.50% | 249.32 |

真实历史相对current在16px容差救回17帧、损害3帧，净增6.39个百分点；相对参数量相同的repeat救回22帧、损害4帧，净增8.22个百分点。在8px下相对current仅净增2.28个百分点（救回22、损害17），所以更好地选中球附近并不意味着精细坐标同幅改善。逐帧及分组统计保存在 `current_vs_stack_seed0.json`、`repeat_vs_stack_seed0.json`；20个current/stack发生16px正确性变化的目标另存 `current_stack_changed_targets.json`，均位于 `outputs/temporal_probe/`。

**困难条件没有随整体指标一起解决。** current/stack/repeat的easy类PCK@16为81.64%/88.41%/79.23%；8个difficult目标三组均未命中，4个occluded目标只有repeat命中1个，stack并未保留这一例。样本过少，不能据此断言重复帧更善于处理遮挡，也不能把stack的easy增益写成困难球恢复。

current在230帧上始终输出有球：219个正例、11个无球误报。原始存在概率中位数在训练正/负例为0.9891/0.8520，验证正/负例为0.9918/0.9363。这里既有默认阈值0.5不能拒绝无球的问题，也有负例跨比赛分布差异；不能未经实验就断言阈值调整或者更强motion一定解决。

stack正确拒绝了1个无球帧，仍有10/11个无球误报；repeat仍有11/11个。三组正例存在召回都是100%。完整检测F1有所改善，存在判别仍基本未解决，不能用约97.6%~97.8%的存在F1掩盖类别不平衡。

按固定规则选取的2个最小和4个最大easy类误差图位于 `outputs/temporal_probe/{current,stack}_seed0/error_examples.png`。本地读图可见大误差对应球场白线凸起、远端人物/器材亮点；真实球部分呈长拖影。stack救回了current在Clip1/0016、0024的两次远端误选，仍在其他帧出现白线/人物亮点竞争。图仅解释具体样例，不代表全部错误频率或背景类别的自动标注。

**精度实测。** `check_probe_precision.py` 现按目标窗口重建输入，一次提取690个唯一验证输入帧，在相同的230个目标上比较三个固定头的重新提取float32特征与实际float16缓存。三组均为0个空间argmax变化、0个0.5存在判断变化；最大存在概率差current/stack/repeat分别约0.0000363/0.0002304/0.0000380。缓存分支精确重现各运行保存的位置指标。该比较包含重新提取的数值差异，不声称隔离了纯量化效应，更不证明两种精度训练轨迹相同。结果为 `outputs/temporal_probe/precision_seed0.json`；此前4目标/12输入的精度smoke也已通过。

```bash
python scripts/check_probe_precision.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --runs outputs/temporal_probe/current_seed0 outputs/temporal_probe/stack_seed0 outputs/temporal_probe/repeat_seed0 --output outputs/temporal_probe/precision_seed0.json
```

**效率边界。** 头训练峰值已分配显存current为108.38MiB，stack/repeat均268.19MiB。时间为冻结缓存训练，不是连续视频端到端延迟；共享GPU环境下也不作精确吞吐排名。repeat在功能上仍只含当前图像，参数重复会改变优化过程，不把其较差结果解释成当前视觉证据变少。

## 依据结果采取的下一步

### seed1、2的实际复核

复用了同一缓存，没有重新解码或提取特征。额外六组的配置记录为 `77003b0` 或 `e1ce6a4`；期间仅增加独立诊断和文档，训练实现仍为 `9e8b271` 的同一版本。完整记录为 `outputs/temporal_probe/seeds012_summary.json`，每个seed的净变化文件也已保存。

| 输入 | seed0 PCK@16 | seed1 PCK@16 | seed2 PCK@16 | PCK@16 均值±样本SD | PCK@8 均值±样本SD |
|---|---:|---:|---:|---:|---:|
| current | 77.17% | 76.71% | 75.34% | 76.41% ± 0.95个百分点 | 59.97% ± 4.09个百分点 |
| stack | 83.56% | 84.02% | 84.02% | 83.87% ± 0.26个百分点 | 65.30% ± 2.54个百分点 |
| repeat | 75.34% | 78.08% | 76.26% | 76.56% ± 1.40个百分点 | 59.21% ± 4.12个百分点 |

stack相对current在三个seed的16px救回/损害分别为17/3、22/6、25/6，净增6.39/7.31/8.68个百分点；相对repeat分别22/4、21/8、22/5，净增8.22/5.94/7.76个百分点。三个seed方向一致，支持在这个固定开发比赛上继续研究真实历史视觉；样本SD只描述优化随机性，不是跨比赛置信区间，也没有把重复预测当作新增独立视频样本。

困难/遮挡仍未稳定恢复：stack仅seed1命中1/8个difficult，所有seed都未命中4个occluded；current和repeat也只在个别seed命中1例。无球方面，九次运行只有stack seed0拒绝1/11个无球目标，其余全部误报；该问题没有因多seed结果稳定而消失。数值精度对照只实际覆盖seed0的三个固定头，不扩写成九个头都已验证。

### 由这些证据作出的选择

真实历史提供了超过两个空间对照的净收益，值得继续，但它证明的是历史视觉输入的价值，未证明已学到可靠correspondence、物理速度或相机运动分离。后续需要更强运动基线与新比赛/球种，不能只在这一个验证比赛继续叠模块。

[独立GT query对应诊断](2026-09-10-correspondence-probe.md)已完成：有限局部范围内有可用的对应，但全局竞争明显降低实际匹配；扩大范围并不单调有益。此处GT只用于诊断，不作为部署输入或自动检测成绩。下一步以已有机制的dense局部cost volume作最小自动定位基线，先区分新增非线性特征与真实跨帧对应的贡献，不预先将其包装成新motion架构。
