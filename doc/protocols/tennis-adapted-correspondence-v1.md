# 前缀适配后的GT-query对应诊断

状态：2026-09-10锁定，真实单查询smoke与四组正式诊断已完成；结果见[实验记录](../experiments/2026-09-10-adapted-correspondence.md)。它是已完成适配实验的机制测量，不增加训练或改选checkpoint。
依据：[前缀适配结果](../experiments/2026-09-10-prefix-adaptation.md)、[原GT-query诊断](tennis-correspondence-probe-v1.md)与[表示解释边界](../literature/2026-09-10-subpixel-correspondence.md)。

## 问题与判别

前缀微调的严格自动定位PCK8在三个seed改善，但检测收益不一致。下一测量回答：在固定当前GT查询格、固定历史候选定义下，选择出的微调前缀是否也改善跨帧匹配排序？若没有，不能把此前的定位收益归因于correspondence；若有但检测不一致，也只说明条件性对应信号与完整检测之间仍有差距。

这个诊断不检验新motion模块，不重启cost宽度、学习率或训练网格，也不替代正在运行的全量竞争系统。旧checkpoint由同一game7按PCK选出，本次再读game7属于开发诊断；不是独立验证或对应任务的最优checkpoint选择。

## 固定输入与模型

使用已有outputs/correspondence_probe/stage1_cosine.json中的逐帧GT-query定义及原因果RGB缓存data/cache/tennis/rgb_512x288_step8_h2。用真实game/clip/frame身份定位同一帧，不改帧号或另读原图。Δ1/Δ2双端合法位置帧对分别215/213，其中双端VC1分别199/190；以实际载入参考的计数为记录依据。

对照为frozen_seed0最终保存的前缀；三个frozen运行均保持官方前缀不更新，因此不重复三次相同视觉forward。比较finetune_seed0、finetune_seed1和finetune_seed2各自已选中的best.pt。只从checkpoint加载prefix参数及ImageNet mean/std，不执行定位头，也不改变原checkpoint。

所有模型都从RGB在线CPU float32计算192×36×64前缀特征，不用旧float16特征与新float32特征混比。每次处理一个当前查询及本次合法历史所需的帧；同一当前查询的Δ1/Δ2共用当前特征，处理完释放，不生成新的大特征缓存。CPU使用2线程，GPU留给既有全量训练。实际先做一个当前查询的小规模运行，确认完整入口、形状、有限输出和成本，再完成四个固定前缀的全部测量。

## 匹配与汇总

复用原analyze_pair、summarize及summarize_groups，不另写cosine、候选搜索、坐标转换或并列排序。该helper原样给出raw与去每帧/每通道空间均值的cosine，以及global、R2/R4/R8搜索；保存这些既有输出不代表重新按结果挑搜索范围。

主要事先指定为去空间均值的Δ1/R2、Δ2/R4历史位置PCK16，以及global精确GT格recall@1/5/10。其余原helper输出作完整辅助记录，不分别挑每个seed的最佳预处理/半径。匹配位置仍在native36×64格回原1280×720坐标，不能使用读出头72×128输出格冒充特征分辨率。

主要解释限制在双端VC1，并分开同格/跨格条件。保留每种(current visibility, history visibility)组合的计数和原指标；当前VC2/历史VC1可作困难条件诊断。任一端VC3只记录标签坐标处的响应，不将其解释为正视觉对应。保留原all聚合便于追溯，但它是合法坐标样本混合统计，不作为已观测视觉对应质量的主要证据。

GT格覆盖与PCK不同：相邻格中心也可能落入16px容差。并列rank及recall沿用原helper：optimistic rank只数严格更高分，实际argmax/top-K按较小格索引破并列。它不是自动发现、稠密光流、camera motion分离或当前球位置预测。

## 输出、验证与后续

新增一个小型CPU入口scripts/probe_adapted_correspondence.py，接收已完成run、已有参考JSON与输出路径，从run配置定位RGB；现有训练及匹配helper保持不变。保存实际代码版本、checkpoint路径及原run配置、CPU线程/精度、输入参考、逐帧结果、分组与耗时，输出位于outputs/adaptation_correspondence/。

验证针对本次新连接：参考与RGB来源不符时不能继续，prefix checkpoint应严格加载，真实小批量需得到预期特征和有限诊断，正式结果需覆盖原参考身份及帧对数量。原cosine/时间接线已有测试，不因新增调用者反复审计数据或重做原图解码。任何失败先修复，未通过入口不开始四组正式诊断。

正式测量不重新训练，也不以结果改变全量DINO的已锁定条件。若匹配变化不能解释自动定位收益，收窄机制表述；即使三个微调前缀均改善该条件指标，仍需自动查询、存在判断与独立来源证据，不能提前写成新的motion representation贡献。

匹配改善也可能来自球/背景外观更可分。本轮没有单帧微调的训练对照，不能据正结果将变化归因于特定时序学习机制；它只能说明已有适配前缀的条件性匹配能力发生了什么变化。
