# 精细定位的适配收益是否伴随匹配排序改善？

日期：2026-09-10。状态：四个固定前缀的CPU诊断与配对汇总完成；本项测量结束，不扩展训练或搜索网格。
协议：[前缀适配后GT-query诊断v1](../protocols/tennis-adapted-correspondence-v1.md)。

## 问题与范围

[已完成的前缀适配](2026-09-10-prefix-adaptation.md)中，微调在三seed都提高严格位置PCK8，但未一致提高检测F1。这个后续只读测量区分“最终定位可读性提高”与“条件性跨帧匹配也改善”，不增加训练、不重新选checkpoint，也不改变全量DINO条件。

使用frozen_seed0及finetune_seed0/1/2最终保存的前缀。三个冻结臂均不更新前缀，故只算一次共同冻结表示。输入为原step8目标对应的真实连续RGB，所有模型都在CPU在线算float32特征；旧float16缓存仅作为参考身份的来源，不作为一臂的特征输入。

当前GT query位于native36×64格，预测历史位置。主要条件为双端VC1，主要匹配固定去空间均值的Δ1/R2、Δ2/R4及global精确格recall@1/5/10。其他原helper结果完整保存但不重新择优。VC3响应不作为正视觉对应；这也不是自动球发现或新motion算法的性能表。

## 最小实现与入口验证

scripts/probe_adapted_correspondence.py从run配置定位RGB，从best.pt严格加载prefix参数与归一化buffer，复用analyze_pair、summarize和summarize_groups。每个当前query只处理当前及需要的唯一历史图像，两个间隔复用当前特征；不执行定位头、不存新的大特征缓存、不读原图、不占用GPU。

新增分组只是在原摘要上补充双端VC1的同格/跨格，以及源visibility组合。来源引用或特征形状不符合固定输入时明确失败；现有时间和标签接线不重复审计。

实际smoke使用原参考JSON中game7/Clip1/8的两个帧对，历史分别为7和6。输入参考另存outputs/adaptation_correspondence/smoke_reference.json，注明源参考与smoke属性，没有修改原诊断。

```bash
python scripts/probe_adapted_correspondence.py --run outputs/adaptation_probe/frozen_seed0 --reference outputs/adaptation_correspondence/smoke_reference.json --output outputs/adaptation_correspondence/smoke.json
```

运行使用Conda zshihyc、CPU2线程；strict加载成功，特征为有限float32 192×36×64。保存1个query、2个pair，Δ1/Δ2身份分别为8→7、8→6，checkpoint epoch0；双端VC1分组与同格/跨格分组计数一致。索引、forward、匹配及汇总共3.24秒，其中归一化与prefix forward为3.07秒。计时不含初始metadata/checkpoint读取、模型构造、导入或最终JSON写入；首批成本不直接外推全部运行或GPU吞吐。

该smoke发生在新脚本提交前，记录的父提交4e0a516不包含新入口；正式诊断在实现提交后启动并另记实际revision。旧匹配算法的测试已覆盖向量对应、候选范围和并列规则，本次未重复跑其原数据或训练。

独立审阅确认来源、raw frame身份、prefix与归一化参数、native格、VC1/VC3分层、固定checkpoint选择及计时边界符合协议，无剩余问题。正式参考共有216个不同当前query，两个间隔共享当前特征；脚本语法与差异格式检查通过。

## 正式运行的固定内容

正式参考为outputs/correspondence_probe/stage1_cosine.json，含215/213个Δ1/Δ2合法坐标帧对，双端VC1为199/190。从提交711a38d顺序完成frozen_seed0、finetune_seed0、finetune_seed1、finetune_seed2；结果保存outputs/adaptation_correspondence/<run>.json，同名.log记录进度。命令模板如下：

```bash
python scripts/probe_adapted_correspondence.py --run outputs/adaptation_probe/frozen_seed0 --reference outputs/correspondence_probe/stage1_cosine.json --output outputs/adaptation_correspondence/frozen_seed0.json
```

每组保存原run配置、实际checkpoint epoch、参考身份、代码版本、CPU条件、逐帧结果、分组和耗时。复用这些JSON作配对比较，没有重新forward或按新指标另选checkpoint。此前自动定位/存在指标直接引用适配实验的保存预测，不混用不同目标范围的分数。

| 前缀 | 已选epoch | CPU索引/前缀/匹配/汇总秒 | 其中归一化与prefix秒 | 结果记录版本 |
|---|---:|---:|---:|---|
| frozen_seed0 | 0 | 602.97 | 584.30 | 711a38d |
| finetune_seed0 | 5 | 469.87 | 458.90 | 711a38d |
| finetune_seed1 | 2 | 276.63 | 269.72 | 711a38d |
| finetune_seed2 | 2 | 374.71 | 363.50 | 1185875 |

诊断入口在四组期间没有变化。最后一组结束时HEAD已包含另一路DINO验证帧复用提交1185875，脚本按结束时HEAD保存版本；该提交未改本诊断及其匹配helper。CPU计时受同期任务与缓存状态影响，不用这几组时间推论模型效率差异。计时边界仍排除初始读取、构造和最终写入。

## 固定主要指标

以下均为**当前GT查询、双端VC1、去空间均值cosine**，不是自动球检测。局部匹配取历史位置的16px容差；global recall要求准确native格，两者数值不能直接比较。几何覆盖在四个前缀之间完全相同：Δ1/R2为196/199，Δ2/R4为187/190。

| 间隔/局部半径 | 前缀 | 局部PCK16 % | global R@1 % | global R@5 % | global R@10 % |
|---|---|---:|---:|---:|---:|
| Δ1/R2，n=199 | frozen | 86.43 | 55.28 | 63.82 | 66.83 |
| 同上 | tuned seed0 | 86.93 | 55.78 | 68.84 | 71.36 |
| 同上 | tuned seed1 | 86.93 | 55.78 | 65.83 | 72.36 |
| 同上 | tuned seed2 | 87.94 | 52.26 | 67.34 | 71.86 |
| Δ2/R4，n=190 | frozen | 70.53 | 38.42 | 53.16 | 59.47 |
| 同上 | tuned seed0 | 77.89 | 42.63 | 57.89 | 65.26 |
| 同上 | tuned seed1 | 75.26 | 42.63 | 55.79 | 64.21 |
| 同上 | tuned seed2 | 74.21 | 42.63 | 57.89 | 61.58 |

局部PCK16的Δ1净增加为1、1、3对，Δ2为14、9、7对。global R@1的Δ2三组均净增加8对，但Δ1为+1、+1、−6；不能概括为所有匹配排序一致改善。global R@5/10上升也不表示一个实际多候选检测器已经获得收益。

![适配后条件性局部匹配与全局排序](../../outputs/adaptation_correspondence/adapted_correspondence.png)

独立PDF位于[同名图表](../../outputs/adaptation_correspondence/adapted_correspondence.pdf)。图只展示固定主要指标，已目视检查标注、范围与表格对应；完整数值保留在[summary.json](../../outputs/adaptation_correspondence/summary.json)。

## 同格与跨格：改善不是所有情况一致

这里“同格/跨格”仅指两个中心映射到36×64 native格是否相同。相同格仍可发生格内运动，跨格也不等于大位移或高rho。

| 间隔与分组 | n | 局部PCK16：frozen / s0 / s1 / s2 % | global R@1：frozen / s0 / s1 / s2 % |
|---|---:|---|---|
| Δ1，同格 | 52 | 96.15 / 94.23 / 96.15 / 96.15 | 80.77 / 75.00 / 73.08 / 75.00 |
| Δ1，跨格 | 147 | 82.99 / 84.35 / 83.67 / 85.03 | 46.26 / 48.98 / 49.66 / 44.22 |
| Δ2，同格 | 21 | 90.48 / 90.48 / 95.24 / 85.71 | 61.90 / 66.67 / 71.43 / 61.90 |
| Δ2，跨格 | 169 | 68.05 / 76.33 / 72.78 / 72.78 | 35.50 / 39.64 / 39.05 / 40.24 |

Δ2的主要局部收益来自跨格：救回/破坏分别16/2、12/4、14/6。同格只有21对，变化很小且方向不一致。Δ1同格的global R@1三组都下降；seed2跨格也下降，尽管其局部PCK16提高。这直接限制“微调使descriptor普遍更可靠”的表述。

Δ1和Δ2既有不同样本数，也使用不同固定搜索半径。可以说这两个既定条件下Δ2净收益较大，不能据此把差异单独归因于时间间隔、速度或大位移能力。

## 配对变化与clip来源

| 主要条件 | s0救回/破坏 | s1救回/破坏 | s2救回/破坏 |
|---|---|---|---|
| Δ1/R2，局部PCK16 | 4 / 3 | 3 / 2 | 5 / 2 |
| Δ2/R4，局部PCK16 | 16 / 2 | 13 / 4 | 15 / 8 |
| Δ1，global R@1 | 8 / 7 | 10 / 9 | 6 / 12 |
| Δ2，global R@1 | 11 / 3 | 10 / 2 | 12 / 4 |

Δ2的190对来自9个clip，其中Clip4占96对。三个微调前缀在Clip4的局部净增加分别11、8、7对；其余94对合计仅+3、+1、0。也就是说，seed2的总体局部净收益全部来自Clip4，seed0/1也主要由它贡献。不能把三个seed在同一来源中的重复改善视为三份跨场景证据。

global Δ2在Clip4以外仍净增加3、3、4对，但只有少量改变，不足以证明跨比赛泛化。Δ1 seed2的global净下降6对，其中5对来自Clip4。保留所有clip的救回与破坏，不只展示获益片段；详细分布在summary.json的by_clip中。

汇总脚本为outputs/adaptation_correspondence/summarize_results.py。已核对四组完整帧对身份、原visibility与query/历史格一致；固定215/213对、双端VC1与同格/跨格计数正确，配对净变化等于聚合指标差，clip计数加总一致。脚本一次运行生成JSON、PNG与PDF；没有再算特征。

## 判断边界

在这项固定开发诊断中，适配后的局部对应确有改善，尤其Δ2跨格条件；不能继续写成“只看到自动位置改善，尚未测量对应”。但global排序并非普遍提高，局部净收益又高度集中于Clip4，因此结论只覆盖这些实际设置。

结合原适配实验，严格自动PCK8三seed改善，但检测F1的seed1/2没有同步提高。这说明**条件性匹配改善与自动检测改善并不等价**。它不证明两者毫无因果关系，也没有证明当前系统缺失一个匹配模块：训练只有位置监督，前缀本身逐帧执行，改进还可能来自球/背景外观可分性或空间读出适配。

本轮没有单帧微调训练对照，GT query也绕过了自动发现；checkpoint在同一game7上按原定位指标选出，本测量属于开发诊断。三个seed不增加独立比赛，VC3响应不作为正视觉对应。这些边界使结果尚不能支持新的motion representation贡献或论文主表。

独立研究审阅核对新汇总与已有自动结果后，支持结束本诊断；特别要求将“三seed改善”限制到共同冻结前缀、同一game7、双端VC1的固定局部PCK16。审阅指出的Δ1同格全局退步、Clip4集中性、检测未同步及同一验证比赛选优边界均已纳入，未要求新的训练或重复数据审计。

本项诊断结束，保存四组结果和最小汇总，不新增单帧适配、半径网格或cost结构。下一判断继续依赖全量HRNet/DINO共同任务结果：先确定剩余错误发生在自动发现、细位置、存在判断还是有视觉证据的对应阶段，再决定是否需要新机制。
