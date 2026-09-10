# 显式局部对应能否改善真实三帧定位？

日期：2026-09-10。状态：seed0筛选、预定seed1/2复核、hidden64控制、全部九个新增cost固定头的精度对照与机制诊断完成。
协议：[Tennis dense局部对应基线](../protocols/tennis-local-cost-probe-v1.md)。

## 假设与判别依据

[三帧输入](2026-09-10-causal-stack.md)已经在三个seed稳定优于当前帧及重复当前帧；[GT query诊断](2026-09-10-correspondence-probe.md)说明局部去均值描述子具有一定匹配能力。本轮检验该证据能否转化为自动定位增益，而非只在给定球位置时有效。

新增raw、centered和self_centered三个cost对照，均保留真实三帧appearance。self只将额外cost的两个历史key替换为当前帧，控制新增相关非线性与参数。局部volume具有明确先例，见[近邻文献](../literature/2026-09-10-local-correspondence-baselines.md)，不作为创新声明。

## 实际实现与smoke

在`08b8244`基础上的本轮修改，新增`ballmotion.correspondence.cost_volume`与缓存入口；复用原训练/评价代码。每个目标在全部36×64格生成106个offset通道，不读取GT query或以visibility筛选。每帧appearance单独GN，cost维持cosine及越界-2的尺度；absence线性层只读appearance。

继续使用Conda `zshihyc`、现有PyTorch及RTX5070Ti Laptop，不安装新依赖。真实smoke源为`data/cache/tennis/temporal_smoke_512`，8个目标包含24个唯一输入帧；直接复用已提取特征。新的cost smoke保存于`data/cache/tennis/local_cosine_smoke/{raw,centered,self_centered}`，每组shape为8×106×36×64。

三种cost一次生成合计1.95秒、峰值已分配显存1152.66MiB；包括源mmap读取、H2D、计算、有限值检查、float16写回与flush。每种元信息中的elapsed为三组共同总时长，不可相加。首批初始化计入第一个raw计算，不能据此认为raw计算比centered本质更慢。每组首次批次转换最大绝对误差为0.000244140625。

已知192维向量测试覆盖历史帧/offset通道、边界哨兵、去均值scalar cosine及self控制。新增head断言证明改变cost不改变appearance归一化或直接改变absence logit；时间拼接测试补充了cost位于完整appearance之后的断言。5项相关时间测试通过，原有8项空间测试及新的1项对应算子测试也已通过。

三个head都完成2epoch、batch4、train4/val4训练和预测保存，参数量均为23,589。输出为`outputs/correspondence_probe/smoke_{raw,centered,self_centered}`；这些极小样本运行只验证通路，位置指标不进入性能比较。

`outputs/correspondence_probe/precision_smoke.json`在4个验证目标上固定同一float16源appearance，比较float32重算cost与实际float16 cost缓存。三组最大cost差均为0.000244140625；位置argmax和0.5存在判断均无变化，缓存分支精确重现保存的位置指标。需要12个唯一源特征，但**重新解码/提取帧数为0**。这只隔离cost存储误差，不证明去均值描述子在float32 backbone与float16 backbone缓存间等价。

精度脚本旧分支因本轮条件执行改动，另用现有三组时序head smoke进行4目标回归：仅重新提取12个输入帧，缓存指标均重现，位置/存在判断无变化。结果为`precision_appearance_regression.json`；没有重复运行完整验证集的旧精度检查。

```bash
python scripts/cache_tennis_correlations.py --source-cache data/cache/tennis/temporal_smoke_512 --output data/cache/tennis/local_cosine_smoke --batch-size 4
# 下列训练对raw/centered/self_centered分别实际执行，使用同样参数。
python scripts/train_spatial_probe.py --cache data/cache/tennis/temporal_smoke_512 --cost-cache data/cache/tennis/local_cosine_smoke/centered --output outputs/correspondence_probe/smoke_centered --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input stack --epochs 2 --batch-size 4
python scripts/check_probe_precision.py --cache data/cache/tennis/temporal_smoke_512 --runs outputs/correspondence_probe/smoke_raw outputs/correspondence_probe/smoke_centered outputs/correspondence_probe/smoke_self_centered --output outputs/correspondence_probe/precision_smoke.json
```

## 完整缓存与正式运行

正式实现版本为`a5a1725`，独立只读审阅未发现需阻止实验的问题；相关6项CPU测试和语法编译通过。缓存位于`data/cache/tennis/local_cosine_s1_h2_r2_r4/{raw,centered,self_centered}`。三个完整1,733×106×36×64 cost共约2.37GiB，一次生成合计60.34秒，batch4、峰值已分配显存1152.66MiB。raw/centered/self的cost_volume GPU累计计算分别4.86/4.52/4.49秒，其余时间包含读取、传输、检查、写回与flush；帧解码及backbone forward均为0。

三组seed0依次完成30epoch，输出为`outputs/correspondence_probe/{raw,centered,self_centered}_seed0`，各有config、history、console、checkpoint和逐帧预测。raw记录版本`a5a1725`，centered/self记录`c64f008`；期间仅增加事后位移统计及文档，训练前向、损失与选优未改变。汇总为`seed0_summary.json`。既有stack来自`9e8b271`的同一数据与读出协议。

```bash
python scripts/cache_tennis_correlations.py --source-cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --output data/cache/tennis/local_cosine_s1_h2_r2_r4 --batch-size 4
# 按raw、centered、self_centered依次实际执行，默认30epoch、batch16。
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --cost-cache data/cache/tennis/local_cosine_s1_h2_r2_r4/centered --output outputs/correspondence_probe/centered_seed0 --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input stack --seed 0
```

## seed0结果与判断

评价使用共同230目标，其中219有位置、11无球。表中的train成绩同样来自验证集选出的checkpoint。

| 模型 | 最佳epoch | train PCK16 | val PCK8 | val PCK16 | val PCK32 | 检测F1@16 | 训练及末次评价秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stack | 8 | 96.63% | 64.84% | 83.56% | 85.39% | 81.70% | 267.84 |
| +raw | 4 | 85.08% | 62.56% | 83.11% | 85.84% | 81.07% | 443.47 |
| +centered | 9 | 97.94% | 57.99% | 85.84% | 88.13% | 84.49% | 399.70 |
| +self_centered | 4 | 88.38% | 62.10% | 82.65% | 84.47% | 80.98% | 867.92 |

![不同误差容差下的seed0定位表现](../../outputs/correspondence_probe/seed0_accuracy.png)

另有[PDF矢量图](../../outputs/correspondence_probe/seed0_accuracy.pdf)。连线只连接已测的8/16/32px结果，不代表未测容差。

三种新增头参数均23,589、峰值已分配显存390.22MiB；stack为20,197和268.19MiB。这些时间来自共享GPU环境和冻结缓存读出，包含输入准备；不是视频端到端速度，不能由self耗时直接推出其模型计算更多。

centered相对stack在16px救回10、损害5，净增5/219；相对self救回7、损害0，净增7/219。它达到事先设定的双对照门槛，因此补固定seed1、2的centered和self，共四次；复用已有三个seed的stack结果，不重训stack、不再生成特征或cost。**这只是进入复核的条件，还不能宣布稳定的motion贡献。** raw相对stack救回5、损害6，未提供净增益，不扩展raw多seed或self_raw。

收益伴随明确的精细定位代价：centered相对stack在8px救回17、损害32，净减15/219；相对self救回11、损害20。中位误差从stack的6.04px变成6.96px；较大错误减少不代表全尺度定位都更好。

8个困难、4个遮挡目标在16px，三种新头均未命中。raw对11个无球全误报；centered/self各误报10个，但另漏判3/1个有位置目标，stack没有这些正例漏判。centered的3个漏判都在Clip7，原帧0016/0112为VC2，0072为VC1，不能把它们统一说成“不可见时合理拒绝”。完整检测成绩与条件PCK分别保留。

逐帧比较为`stack_vs_{raw,centered,self_centered}_seed0.json`和`self_centered_vs_centered_seed0.json`。辅助位移分组在完整结果产生前固定：Δ1同格/范围内移动/范围外分别57/155/3个双端合法目标，Δ2为22/187/4。centered相对stack在范围内移动组16px净增1/155、3/187，相对self净增5/155、6/187；范围外均无净变化且样本过少。这不能单独支持远范围搜索的收益主张。

位移分组的已知点测试覆盖救回、新错、无配对样本及空组；新增2项对应测试通过。复用旧current/stack预测的集成结果精确保持既有全集paired统计，证据为`current_vs_stack_motion_seed0.json`。

## 完整精度对照与传输开销

`precision_seed0.json`覆盖三个完整固定头的全部230个验证目标，使用690个唯一源特征，重新解码/提取帧数仍为0。固定同一float16 appearance，float32重算cost与实际float16 cost的最大差均为0.000244140625；三个头均无空间argmax或0.5存在判断变化，缓存分支精确重现保存的位置指标。raw/centered最大logit差为0，self为0.0000166893；不把这些结果扩写为其他骨干数值设置或两种精度的训练轨迹等价。

```bash
python scripts/check_probe_precision.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --runs outputs/correspondence_probe/raw_seed0 outputs/correspondence_probe/centered_seed0 outputs/correspondence_probe/self_centered_seed0 --output outputs/correspondence_probe/precision_seed0.json
```

self训练明显变慢后，一次资源快照未发现持续大量换页；确认另一个GPU任务来自`event`项目，未调整它。随后单独测量同一真实batch16、682×36×64的CPU float16输入：合并传输/转float32的`.to(cuda, float32)`中位54.08ms；先传float16再GPU转float32的`.to(cuda).float()`中位8.47ms。两条路径交替顺序各测7次，最终张量逐项完全相同；此段显存峰值从96增至144MiB。证据为`transfer_benchmark.json`，不能将这一局部约6.4倍差异宣称为整体训练加速。

因此仅修改训练和预测两处传输表达式，保持进入模型的float32数值不变，供后续seed复核使用；不更改网络、输入、损失或选优规则，也不重跑已完成seed0。这是实际瓶颈的一项减少，不代表已解释self耗时的全部变化。

## 错误候选的高相关值：辅助机制诊断

这是探索性诊断，未据此训练新模型或选取阈值。固定旧stack seed0的错误目标（原图误差>16px且错误峰不在GT同一native cell），只取GT query JSON中双端均VC1的配对；比较**当前GT格**与**当前错误峰格**的centered cost，Δ1/R2有21对，Δ2/R4有20对。错误峰不是人工标注的“背景类别”，标签位移也没有分离相机运动。

对每个当前格分别计算：`max_delta C`，以及`max_(delta!=0) C - C(delta=0)`。后一项只表示非零偏移相对于零偏移的匹配优势，不是物理motion reliability。

| 时距 | 错误峰最大cos高于GT格 | GT非零偏移优势高于错误峰 | 最大cos中位数：GT/错误峰 |
|---|---:|---:|---:|
| Δ1 | 14/21 | 15/21 | 0.8003 / 0.9238 |
| Δ2 | 12/20 | 13/20 | 0.7642 / 0.8552 |

简单最大相关值会偏向许多已有错误候选，因此不能直接把它当作“这个位置是球且motion可靠”的置信度。非零偏移比较在这批样本更有区分力，但仍有6/21、7/20反例，且样本条件化、来源少；不能据此宣布一个可靠性模块成立。下一设计若压缩offset volume，必须注意保留相对零位移的结构，不能未经检验只压成一个最大值。

全过程只读既有三帧特征索引、`stage1_cosine.json`、stack预测CSV及centered cost，未做解码/GPU计算。每对的身份、两个native cell和原始分数保存在`outputs/correspondence_probe/cost_peak_diagnostic.json`。它不直接证明新head实际如何使用cost，也不是所有空间点的无条件统计。

## 细定位退步发生在哪里？

固定每个模型实际选中的36×64 native block，只在该block对应的四个72×128输出子格里，使用GT选择最近一个。这是诊断性的条件oracle，不改变预测，也不等于整个模型的可达上限。先重现原始PCK8，再检查oracle误差不大于实际误差，以排除坐标或块映射错误。

| seed0头 | 实际PCK8命中 | 固定预测块的四子格oracle命中 | 块内存在8px解但未选中 | 预测块等于GT native块 |
|---|---:|---:|---:|---:|
| stack | 142/219 | 167/219 | 25 | 148 |
| raw | 137/219 | 173/219 | 36 | 154 |
| centered | 127/219 | 167/219 | 40 | 143 |

stack与centered选中同一native block的162个目标中，centered在8px救回5、损害17；block不同的57个目标中救回12、损害15。因此主要净损失发生在两者粗块相同的样本，不能只解释为远处背景误选，也不能把16px提升说成“正确native块选得更多”。GT恰好位于网格边缘时，相邻块也可能包含8px内的输出点，所以“GT同块数”与位置容差oracle本就不同。

现有证据与“native cost经共享hidden32影响四个PixelShuffle子格的选择”一致，但没有证明其因果机制；新增非线性、优化随机性和checkpoint选优同样可能参与。centered按PCK16选中的epoch9 PCK8为57.99%，而epoch3曾为62.56%，说明选优规则解释部分退步；不能在事后换成按PCK8选优再声称本轮主要比较变好。

输出为`outputs/correspondence_probe/subcell_diagnostic_seed0.json`。独立只读研究审查认为当前诊断已足以区分粗块选择与块内选择，不应继续凭单seed加模块；先完成self与预固定seeds条件。

## 预定三seed复核：小幅粗定位收益，持续细定位损失

centered和self_centered各seed1、2均完成30epoch，使用版本`72ee2d9`。这一版本只调整前述输入传输表达式，不改变网络数值、损失或选优规则；四次运行继续复用同一冻结特征和cost缓存。已有stack三seed作为对应控制，不重复训练。以下汇总采用同一game7的230个目标，其中219个有合法位置。

| 模型 | seed | 最佳epoch | PCK8 | PCK16 | PCK32 | 检测F1@16 |
|---|---:|---:|---:|---:|---:|---:|
| stack | 0 | 8 | 64.84% | 83.56% | 85.39% | 81.70% |
| centered | 0 | 9 | 57.99% | 85.84% | 88.13% | 84.49% |
| self_centered | 0 | 4 | 62.10% | 82.65% | 84.47% | 80.98% |
| stack | 1 | 6 | 63.01% | 84.02% | 86.30% | 81.96% |
| centered | 1 | 1 | 61.64% | 83.11% | 86.30% | 81.07% |
| self_centered | 1 | 1 | 62.56% | 83.11% | 85.84% | 81.07% |
| stack | 2 | 5 | 68.04% | 84.02% | 85.39% | 81.96% |
| centered | 2 | 5 | 64.84% | 85.39% | 87.67% | 83.30% |
| self_centered | 2 | 8 | 65.75% | 82.65% | 84.47% | 81.08% |

| 模型 | PCK8，均值±样本SD | PCK16，均值±样本SD | PCK32，均值±样本SD |
|---|---:|---:|---:|
| stack | 65.30±2.54% | 83.87±0.26% | 85.69±0.53% |
| centered | 61.49±3.43% | 84.78±1.47% | 87.37±0.95% |
| self_centered | 63.47±1.99% | 82.80±0.26% | 84.93±0.79% |

这里SD的数值单位是百分点，只反映同一比赛、同一目标集合下的优化随机性。不是跨比赛置信区间；也不能把三次重复的219个位置当作657个独立验证目标。raw只完成事先规定的seed0筛选，不给它拼凑多seed均值。

![同一验证比赛上三seed的定位结果](../../outputs/correspondence_probe/multiseed_accuracy.png)

[PDF矢量图](../../outputs/correspondence_probe/multiseed_accuracy.pdf)。点为各seed，误差线为均值±样本SD；三个面板使用各自的纵轴范围。

| centered的对照 | seed | 8px救回/新错 | 16px救回/新错 | 32px救回/新错 |
|---|---:|---:|---:|---:|
| stack | 0 | 17 / 32 | 10 / 5 | 9 / 3 |
| stack | 1 | 22 / 25 | 10 / 12 | 8 / 8 |
| stack | 2 | 15 / 22 | 8 / 5 | 8 / 3 |
| self_centered | 0 | 11 / 20 | 7 / 0 | 8 / 0 |
| self_centered | 1 | 1 / 3 | 1 / 1 | 2 / 1 |
| self_centered | 2 | 13 / 15 | 11 / 5 | 10 / 3 |

相对stack，centered的PCK16平均增加0.91个百分点，但逐seed为+2.28、−0.91、+1.37，并未三个seed都优于简单堆叠；PCK8平均减少3.81个百分点，三个seed方向一致。相对self，PCK16为两次增加、一次持平，平均增加1.98个百分点；PCK8三次下降，平均减少1.98个百分点。PCK32比stack两次提高、一次持平，比self三次提高。这支持“跨帧cost在当前读出下有减少一部分较大错误的信号”，不支持“该设计已经稳定改善精确球定位”。

困难和遮挡没有随重复实验改善：centered与self三个seed在VC2的8个目标、VC3的4个目标上PCK16均为0；stack仅seed1在VC2命中1个。新增四次运行中，centered seed1/2和self seed1均对11个无球全部误报，且没有正例存在漏判；self seed2误报10个无球，同时漏判4个有位置目标。条件PCK仍评价全部219个位置，不能用它掩盖存在判断失败。

四次训练及末次评价耗时按centered seed1、self seed1、centered seed2、self seed2分别为216.20、382.76、281.11、183.98秒，峰值已分配显存均391.22MiB。优化后的传输路径没有使所有运行耗时一致；共享GPU的总时间不适合用来判断两种cost的算法复杂度。

```bash
# variant取centered/self_centered，seed取1/2；实际执行四次，全部完成。
python scripts/train_spatial_probe.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --cost-cache data/cache/tennis/local_cosine_s1_h2_r2_r4/centered --output outputs/correspondence_probe/centered_seed1 --stage 1 --output-stride 4 --hidden-channels 32 --temporal-input stack --seed 1
python scripts/check_probe_precision.py --cache data/cache/tennis/dinov3_convnext_tiny_512x288_step8_h2_s1 --runs outputs/correspondence_probe/centered_seed1 outputs/correspondence_probe/self_centered_seed1 outputs/correspondence_probe/centered_seed2 outputs/correspondence_probe/self_centered_seed2 --output outputs/correspondence_probe/precision_seeds12.json
```

新增四个固定头的完整230目标精度比较仍未发现空间argmax或存在判断变化；最大cost差均为0.000244140625，最大logit差不超过0.0000200272。缓存分支逐项重现各自已保存的位置指标，重新解码/提取帧数为0。结合seed0检查，七个新增固定头均已覆盖；结论仍仅限于相同源float16 appearance上的cost存储差异。

完整汇总位于`outputs/correspondence_probe/multiseed_summary.json`，包含每个运行的版本、路径、选中epoch、评价、耗时及三seed paired结果。seed1/2比较沿用已有`compare_predictions.py`和同一GT query位移分组，保存为`{stack,self_centered}_vs_centered_seed{1,2}.json`；未重新运行位置预测来计算paired统计。

## 本轮研究决定

不将当前dense cost拼接升为默认模型，也不据此继续叠加更远搜索、多假设或可靠性模块。保留简单三帧stack作为下一阶段控制，保留centered/self及缓存作为可复用机制基线。对应信号没有被完全否定，但当前融合同时带来更稳定的精细定位损失；需要先分辨表示、读出与训练条件，不能把冻结小头的性能当作现代backbone能力上限。

单seed子格诊断仍是探索性解释，不因三seedPCK8下降就自动升级为因果证明。当前开发集合的范围外目标极少，且只有一个验证比赛；这些实验不能证明大位移问题已经解决。下一阶段优先补足对读出/训练瓶颈有区分力的受控比较，再进入完整强基线与跨球种验证，不以增加模块数量代替证据。

## 后续容量控制：运行前设定

独立只读研究审查建议先使用已有宽度参数，检验hidden32是否限制了融合；不为这个问题先编写新的残差、门控或分支。现已在同一[协议](../protocols/tennis-local-cost-probe-v1.md)追加锁定hidden64、seed0的stack/centered/self三组控制。只有centered64在8px不损害两个同宽度对照、且16px严格优于二者时才补seed1/2。这个控制同时改变容量和优化，不足以单独证明特定神经元竞争机制。

以上为运行前设定；原hidden32全部结果仍有效且保持原始选优条件。当前实验不引入新代码、依赖或特征缓存。

## hidden64实际结果与停止条件

三组以版本`4c91997`完成30epoch，全部按原规则选中epoch4。输出为`outputs/correspondence_probe/width64_{stack,centered,self_centered}_seed0`。

| 同一hidden64 | 参数 | train PCK16 | val PCK8 | val PCK16 | val PCK32 | 秒数 | 峰值已分配MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| stack | 39,813 | 91.88% | 62.10% | 82.65% | 84.47% | 79.84 | 297.86 |
| centered | 46,597 | 92.23% | 59.82% | 85.84% | 89.04% | 113.76 | 400.58 |
| self_centered | 46,597 | 92.92% | 62.10% | 83.56% | 85.84% | 105.39 | 400.58 |

centered相对stack在8px救回15、新错20，在16px救回12、新错5；相对self在8px救回8、新错13，在16px救回6、新错1。它的PCK8仍低于两个同宽度对照各5/219，即2.28个百分点，**没有达到预定复核条件**。因此不补hidden64的seed1/2，也不继续搜索更大宽度或添加cost融合模块。

centered和self在VC2均命中1/8，stack为0；三者VC3均0/4。新增困难命中同样出现在self，不能专门归因于跨帧对应。centered/self各误报10/11个无球，并漏判2个有位置目标；stack误报11个无球但没有正例漏判。

`precision_width64.json`覆盖两个新增cost固定头的完整230目标。float32重算cost与实际float16缓存之间仍为0空间argmax变化、0存在判断变化，最大cost差0.000244140625；最大logit差分别0.0000257492、0.0000438690，缓存位置指标重现保存结果。未解码/重新提取图像。

汇总为`width64_summary.json`，逐帧及位移条件比较为`width64_{stack,self_centered}_vs_centered_seed0.json`。时间仍仅是共享设备上的冻结读出实验，不是完整视频速度。

这次负结果不能证明“读出容量与问题无关”，因为只试了一个宽度和seed；它足以否定当前这个预定的继续条件。三帧hidden32仍作为下一阶段无cost的初始化/控制。先检验相同模型初始化下冻结前缀继续训练与共同微调的区别，再建立完整系统基线；不因局部对应概念符合直觉就保留当前融合方式。
