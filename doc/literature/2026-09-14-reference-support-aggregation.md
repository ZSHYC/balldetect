# 目标帧保留与支撑聚合：哪些机制已经有先例

日期：2026-09-14。性质：围绕下一项融合决策的有界文献补读，不是新模型复现。本轮补齐SELSA和Temporal RoI Align的方法、消融与关键源码；FGFA的补读另见下文。既有[BIRD/STSN全文证据](2026-09-12-task-supervised-alignment.md)和[近期运动文献](2026-09-12-recent-tiny-motion.md)继续复用，不重新检索整个领域。

## 对当前研究的直接影响

“保留当前帧、向支撑帧找多个对应、再晚融合”已经有很近的检测先例，不能仅以这个模块组合主张新颖性。需要区分三个不同操作位置：在稠密图上聚合后再生成候选；在已有目标候选的局部网格上找支撑；在各帧已有候选之间做语义汇聚。它们给真球进入模型计算的机会不同，不能都简称为候选时序attention。

本项目[center5候选与单帧互补诊断](../experiments/2026-09-14-blurball-centered-candidates.md)已经限制了直接回退单帧和固定候选重排的收益；这些文献进一步要求解释聚合之前保留了什么、聚合之后丢掉什么。当前仍先完成[center3长度控制](../experiments/2026-09-14-blurball-centered-length.md)，不因补读论文同时增加分支、框检测器或新损失。

## FGFA：先对齐聚合，再生成候选

FGFA论文中，参考帧是光流/相似度的条件输入，也是包含在中心时间窗口内的聚合项。邻帧特征经双线性warp后，使用嵌入余弦相似度计算逐位置、跨帧归一化权重；聚合特征送入检测器，没有另外一条未融合目标特征的检测头。论文采用21帧推理；无光流的均匀聚合为72.0 mAP，完整FGFA为76.3，单帧基线73.4，说明增加帧并不自动改善定位。[ICCV 2017论文，arXiv v2，§3–4](https://arxiv.org/pdf/1703.10025v2)

作者测试图确认RPN/Proposal接在聚合之后。因此它不是先保留一份单帧候选集合，再用光流给这些候选加分。若融合后候选遗漏了球，下游候选读出没有一份独立单帧候选可以自动兜底；但这不等于证明FGFA在本项目会漏球。[作者固定源码，测试图](https://github.com/msracver/Flow-Guided-Feature-Aggregation/blob/f01243e23a2e2be0a91d0a5a927433686305a054/fgfa_rfcn/symbols/resnet_v1_101_flownet_rfcn.py)

补读同时发现需要保留的实现区别：该提交训练图输入目标及两个支撑，却只对两张warp后的支撑做加权，目标主要作相似度query；目标自身不是无条件加入的第三个聚合项。测试图则包含目标项。测试配置为中心±9、共19帧，不能与论文21帧混写。另一个区别是发布实现预先计算并缓存embedding，再一起warp；不能逐算等同于论文描述的先warp特征再过embedding。[同一训练/测试源文件](https://github.com/msracver/Flow-Guided-Feature-Aggregation/blob/f01243e23a2e2be0a91d0a5a927433686305a054/fgfa_rfcn/symbols/resnet_v1_101_flownet_rfcn.py)、[发布配置](https://github.com/msracver/Flow-Guided-Feature-Aggregation/blob/f01243e23a2e2be0a91d0a5a927433686305a054/fgfa_rfcn/config/config.py)

这里引用源码是为限定真实计算图，不据此猜测作者所有checkpoint的训练过程，也不把论文机制改写成源码某一版本。FGFA仍是reference-conditioned对齐和自适应聚合的明确先例。

**计算复用的实际边界。** 发布测试器用滑窗保存逐帧特征和embedding，每前进一步提取新帧；对每个新reference，成对FlowNet和warp仍要计算。缓存外观不等于缓存了所有时序计算。[作者测试器](https://github.com/msracver/Flow-Guided-Feature-Aggregation/blob/f01243e23a2e2be0a91d0a5a927433686305a054/fgfa_rfcn/core/tester.py) 论文还报告组合相邻flow来降低成本会损失约1点mAP；这提醒本项目，改变跨帧计算复用规则时要验证预测语义，不能把有损运动组合当作免费加速。[论文§4.2](https://arxiv.org/pdf/1703.10025v2)

## SELSA：语义邻居与目标残差，不是物理位移

SELSA先用RPN产生各帧候选，再对RoI特征进行跨候选语义聚合，目标是利用类别相关外观，不要求对应点按时间顺序关联。ImageNet VID训练采样目标加两张同视频随机帧，默认推理从序列采样21帧。消融中的单帧语义聚合为75.26 mAP，跨帧为80.25；增加数据增强后的82.69是另一条件，不能将全部收益归给聚合。[论文v1，2019-07-15，§3–4](https://arxiv.org/html/1907.06390v1)

作者MXNet代码中，两次`semantic_aggregation`分别加回当前层特征后进入ReLU；训练和推理均能看到该残差连接。其相似度softmax随后加权的是支撑候选的value特征。这里确认的是目标候选表征的残差保留，并非独立运行一个单帧检测器或保留一份不可更改的单帧预测。[作者源码，读取于本日](https://github.com/happywu/Sequence-Level-Semantics-Aggregation/blob/master/rcnn_selsa/symbols/resnet_v1_101_rcnn_selsa.py)

MMTracking v0.14.0的SELSA重实现也在每个共享FC后加入聚合残差，然后激活；它的多头投影、维度缩放和实现细节应按该版本记录，不能不加区别地写成作者原始MXNet模型。[重实现聚合器](https://github.com/open-mmlab/mmtracking/blob/v0.14.0/mmtrack/models/aggregators/selsa_aggregator.py)、[重实现box head](https://github.com/open-mmlab/mmtracking/blob/v0.14.0/mmtrack/models/roi_heads/bbox_heads/selsa_bbox_head.py)

**本项目推论。** 用支撑帧改善球类别证据可以有效，但类别相似并不确定当前候选就是球，更不提供帧间位移。残差路径也不数学保证身份不变：学习到的聚合值仍可能改变或抵消目标响应。因此“当前帧有skip”既不是新贡献，也不是模型正确性的保证。

## Temporal RoI Align：局部query、全图搜索、两级聚合

该方法先为目标帧RPN候选抽取7×7 RoI特征；每个网格位置与支撑帧全图特征作余弦相似度，取top‑K位置并加权汇聚。之后将目标自身与各支撑的RoI表示做多头时间聚合。正文使用K=4、4个时间注意块；训练目标加两张随机支撑，默认推理用全视频均匀采样的14张支撑。[论文v1，2021-09-08，方法与实现](https://arxiv.org/html/2109.03495v1)

正文的同一ImageNet VID/R101消融中，单帧基线74.0 mAP、相似检索后直接平均78.5、再用时间attention80.5；K从4继续增加没有继续改善。它给出“检索”和“如何融合”的分别比较，未提供BlurBall几像素中点定位的保证。[同一论文，Tables 1、3](https://arxiv.org/html/2109.03495v1)

**源码进一步明确的事实。** `most_similar_roi_align`先构造query与所有支撑空间格的完整相似度矩阵，再按每个支撑帧独立取top‑K；每个支撑内的K个value被softmax加权成一个向量。目标RoI被拼入时间维，所有帧一起softmax汇聚；最终没有额外加回一份未加权目标RoI，也没有显式no-match输出。[作者链接的MMTracking v0.14.0实现](https://github.com/open-mmlab/mmtracking/blob/v0.14.0/mmtrack/models/roi_heads/roi_extractors/temporal_roi_align.py)

这不是“找了K个地址，就向后保留K条互斥运动假设”。地址在采样value后被消耗，输出仍是按目标RoI网格排列的融合特征；它可以含运动相关信息，但没有直接输出每条候选位移的概率和对应身份。相应地，目标参与加权只提供可选择的自身特征，不保证其权重有可用下界。

### 发布配置与正文不能混用

本次固定读取的MMTracking v0.14.0配置设`num_most_similar_points=2`、3个共享FC、7个训练epoch；R101配置继承该R50配置并替换backbone。不能把运行这份配置称为逐项复现正文的K=4/6epoch设置。源码中的时间attention还使用缩放点积，其具体归一化顺序以实现为准。[基础配置](https://github.com/open-mmlab/mmtracking/blob/v0.14.0/configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py)、[R101继承配置](https://github.com/open-mmlab/mmtracking/blob/v0.14.0/configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid.py)

这里没有复现这些配置，也不因存在差异判定作者结果无效；它只是约束未来采用实现时的来源表述。

## 候选、支撑与拒绝的三项边界

**目标候选与支撑候选不是同一个筛选环节。** SELSA使用各帧已有RoI，Temporal RoI Align的支撑检索则直接读全图，不需要先生成一个准确支撑框；后者仍以目标RoI作为query入口。本项目若采用候选点query，必须测目标候选的漏球，不能仅报告给定真球query后的匹配成功率。box回归还可以移动输出，不能把我们的“固定候选坐标上限”原样套到带回归的检测器。

**先全图检索再top‑K，不等于只计算K个匹配。** 从所读实现直接推导：设R个目标候选，每候选h×w个query、S张支撑、每张H×W空间格、C通道，则相似度计算量量级为`S·R·h·w·H·W·C`，未分块实现的分数张量量级为`S·R·h·w·H·W`。top‑K主要缩减后续value汇聚范围。减少query数量、缩小支撑分辨率和加速计算核分别改变不同成本，不能混成一次“高效匹配”贡献。这是张量形状推导，不是本地速度或显存实测。

**权重不是球身份或无对应概率。** 例如单位余弦分数的top‑K softmax在没有温度缩放时，只描述K个分数的相对关系；它没有“所有候选都不对”的类别。目标帧自身可以提供无跨帧支持时的外观证据，但这与明确判断某个支撑不存在可靠对应仍是两回事。即便今后引入no-match，也需对照[已有null/置信度先例](2026-09-12-native-matching-confidence.md)，不能把它当首次可靠性建模。

## 对后续实验的约束

本项目已经做过冻结候选的[软对应汇聚](../experiments/2026-09-13-blurball-candidate-set-supervision.md)与[多个地址拼接](../experiments/2026-09-13-blurball-candidate-addresses.md)，当时没有取得独立motion增量。本轮文献不能用来把这些负结果改写成阳性，也不支持仅换成RoI术语后重跑相同配方。可能有区分力的改动必须具体到共同训练、聚合前的任务证据、候选准入或压缩发生位置，并保留对应的简单竞争解释。

三篇方法中的视频范围和对象尺度均不能直接继承给本项目：即使无时序递归，全视频采样也可能越过我们的合法rally或固定前瞻范围；框mAP提升更不能替代严格中心误差和输出状态。这里保留可比较的机制来源，不批准全视频输入、自动生成伪球框或额外motion标注。

本轮只读一手文献与关键源文件，没有安装MMTracking/MXNet、运行上游模型或改动正在训练的模型。访问失败的CVF PDF未作为公式核对证据，SELSA机制通过可访问的arXiv全文与作者源码确认；其master源码记录读取日期，未将其声称为固定发布版本。文献与本地实验引用检查不触发数据或训练重验。
